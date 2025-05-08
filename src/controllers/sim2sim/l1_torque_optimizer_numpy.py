import numpy as np
from src.controllers.sim2sim.qp_torque_optimizer_numpy import QPTorqueOptimizer, convert_to_skew_symmetric_batch

class L1TorqueOptimizer:
    """Wraps QPTorqueOptimizer with an L1 adaptive loop."""
    def __init__(self, robot, qp_kwargs: dict, l1_kwargs: dict):
        # Nominal QP solver from CAJun
        import inspect
        sig = inspect.signature(QPTorqueOptimizer.__init__).parameters
        qp_args = {k: v for k, v in qp_kwargs.items() if k in sig}
        self._qp = QPTorqueOptimizer(robot, **qp_args)
        self.dt = l1_kwargs['dt']

        # L1 design gains
        self.L = l1_kwargs['observer_gain']      # observer gain (6x6)
        self.Gamma = l1_kwargs['adapt_gain']     # adaptation rate
        self.omega_c = l1_kwargs['filter_cutoff']

        # Build B matrix: maps 12D GRFs -> 6D wrench
        inv_mass = self._qp._inv_mass         # (3x3)
        inv_inertia = self._qp._inv_inertia   # (3x3)
        p = robot.foot_positions_in_base_frame[0]  # (4x3)
        B_lin = np.concatenate([inv_mass]*4, axis=1)           # (3x12)
        px = convert_to_skew_symmetric_batch(p[None,:,:])[0]   # (3x12)
        B_ang = inv_inertia.dot(px)                           # (3x12)
        self.B = np.vstack([B_lin, B_ang])                    # (6x12)

        # Internal states
        self.x_hat = np.zeros(6, dtype=np.float32)
        self.d_hat = np.zeros(6, dtype=np.float32)
        self.filt_buf = np.zeros(6, dtype=np.float32)
        self.u_ref = np.zeros(6, dtype=np.float32)

        # For finite-difference accel
        self.prev_lin_vel = robot.base_velocity_body_frame.copy()
        self.prev_ang_vel = robot.base_angular_velocity_body_frame.copy()

    def _extract_wrench(self, grf: np.ndarray) -> np.ndarray:
        # grf is a 12-element vector of ground reaction forces
        f = grf.reshape(4,3)
        F = np.sum(f, axis=0)
        p = self._qp._robot.foot_positions_in_base_frame[0]
        M = np.zeros(3, dtype=np.float32)
        for i in range(4):
            M += np.cross(p[i], f[i])
        return np.concatenate([F, M], axis=0)

    def _predictor(self, y_meas: np.ndarray):
        # Predictor: xdot = (u_ref + d_hat) + L (y_meas - x_hat)
        xdot = (self.u_ref + self.d_hat) + self.L.dot(y_meas - self.x_hat)
        self.x_hat += xdot * self.dt

    def _adaptation(self, y_meas: np.ndarray):
        # Adaptation law: d_dot = -Gamma * (x_hat - y_meas)
        err = self.x_hat - y_meas
        d_dot = - self.Gamma * err
        self.d_hat += d_dot * self.dt

    def _l1_filter(self) -> np.ndarray:
        # Discrete 1st-order low-pass: C(s)=omega_c/(s+omega_c)
        alpha = self.omega_c * self.dt / (1 + self.omega_c * self.dt)
        self.filt_buf = alpha * self.d_hat + (1 - alpha) * self.filt_buf
        return self.filt_buf

    def get_action(self, foot_contact_state: np.ndarray, swing_foot_position: np.ndarray):
        # 1) Nominal QP solve
        motor_command, desired_acc, solved_acc, qp_cost, num_clips = \
            self._qp.get_action(foot_contact_state, swing_foot_position)

        # 2) Get raw GRF from QP direct solve
        _, _, grf, _, _ = self._qp._solve_joint_torques(foot_contact_state, desired_acc)
        flat_grf = grf[0] if grf.ndim>1 else grf
        self.u_ref = self._extract_wrench(flat_grf)

        # 3) Measure actual base accelerations
        curr_lin = self._qp._robot.base_velocity_body_frame[0]
        curr_ang = self._qp._robot.base_angular_velocity_body_frame[0]
        meas_lin_acc = (curr_lin - self.prev_lin_vel[0]) / self.dt
        meas_ang_acc = (curr_ang - self.prev_ang_vel[0]) / self.dt
        self.prev_lin_vel[0] = curr_lin.copy()
        self.prev_ang_vel[0] = curr_ang.copy()
        y_meas = np.concatenate([meas_lin_acc, meas_ang_acc], axis=0)

        # 4) L1 predictor + adaptation
        self._predictor(y_meas)
        self._adaptation(y_meas)
        delta = self._l1_filter()

        # 5) Inject correction into desired acceleration
        corrected_acc = desired_acc.copy()
        corrected_acc[0, :3] -= delta[:3]
        corrected_acc[0, 3:] -= delta[3:]

        # 6) Final QP solve with corrected target
        motor_torques, final_solved_acc, final_grf, final_qp_cost, final_clips = \
            self._qp._solve_joint_torques(foot_contact_state, corrected_acc)

        # Reconstruct MotorCommand
        from src.robots.sim2sim.motors import MotorCommand
        motor_command_obj = MotorCommand(
            desired_position=motor_command.desired_position,
            kp=motor_command.kp,
            desired_velocity=motor_command.desired_velocity,
            kd=motor_command.kd,
            desired_extra_torque=motor_torques
        )

        return motor_command_obj, corrected_acc, final_solved_acc, final_qp_cost, final_clips

    # ---- Forwarded properties for compatibility ----
    @property
    def desired_base_position(self): return self._qp.desired_base_position
    @desired_base_position.setter
    def desired_base_position(self, val): self._qp.desired_base_position = val
    @property
    def desired_base_orientation_rpy(self): return self._qp.desired_base_orientation_rpy
    @desired_base_orientation_rpy.setter
    def desired_base_orientation_rpy(self, val): self._qp.desired_base_orientation_rpy = val
    @property
    def desired_linear_velocity(self): return self._qp.desired_linear_velocity
    @desired_linear_velocity.setter
    def desired_linear_velocity(self, val): self._qp.desired_linear_velocity = val
    @property
    def desired_angular_velocity(self): return self._qp.desired_angular_velocity
    @desired_angular_velocity.setter
    def desired_angular_velocity(self, val): self._qp.desired_angular_velocity = val
