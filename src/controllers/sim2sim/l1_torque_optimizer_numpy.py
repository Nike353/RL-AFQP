import numpy as np
from src.controllers.sim2sim.qp_torque_optimizer_numpy import (
    QPTorqueOptimizer,
    convert_to_skew_symmetric_batch,
)

class L1TorqueOptimizer:
    """QPTorqueOptimizer with an L1 adaptive loop."""
    def __init__(self, robot, qp_kwargs: dict, l1_kwargs: dict):
        # — your existing initialization unchanged —
        import inspect
        sig     = inspect.signature(QPTorqueOptimizer.__init__).parameters
        qp_args = {k: v for k, v in qp_kwargs.items() if k in sig}
        self._qp = QPTorqueOptimizer(robot, **qp_args)
        self.qp_log = {
            'grf_nominal': [],  # list of 12‑vector GRFs
            'time': []          # list of timestamps
        }

        self.dt      = l1_kwargs['dt']
        self.L       = l1_kwargs['observer_gain']
        self.Gamma   = l1_kwargs['adapt_gain']
        self.omega_c = l1_kwargs['filter_cutoff']

        inv_mass    = self._qp._inv_mass
        inv_inertia = self._qp._inv_inertia
        p           = robot.foot_positions_in_base_frame[0]
        B_lin       = np.concatenate([inv_mass]*4, axis=1)
        px          = convert_to_skew_symmetric_batch(p[None,:,:])[0]
        B_ang       = inv_inertia.dot(px)
        self.B      = np.vstack([B_lin, B_ang])

        self.x_hat    = np.zeros(6, dtype=np.float32)
        self.d_hat    = np.zeros(6, dtype=np.float32)
        self.filt_buf = np.zeros(6, dtype=np.float32)
        self.u_ref    = np.zeros(6, dtype=np.float32)

        # ==== NEW: set up logging dictionary ====
        self.log = {
            'grf_nominal':   [],   # raw GRFs from the first QP
            'd_hat':         [],   # disturbance estimate
            'delta':         [],   # filtered disturbance
            'grf_corrected': [],   # GRFs from the corrected QP
            'time':          [],   # timestamp (seconds)
        }

        # ==== NEW: automatically dump self.log on exit to a fixed file ====
        import atexit
        self._dump_file = "l1_log.npz"
        def _dump_logs():
            # convert lists to arrays and save
            npz_dict = {k: np.array(v) for k,v in self.log.items()}
            np.savez(self._dump_file, **npz_dict)
            print(f"[L1TorqueOptimizer] logs saved to {self._dump_file}")
        atexit.register(_dump_logs)

        def _dump_qp_logs():
            # convert lists -> arrays
            import numpy as _np
            data = {
                'grf_nominal': _np.array(self.qp_log['grf_nominal']),
                'time':        _np.array(self.qp_log['time']),
            }
            _np.savez("qp_log.npz", **data)
            print("[QPTorqueOptimizer] logs saved to qp_log.npz")
        atexit.register(_dump_qp_logs)

        self.prev_lin_vel = robot.base_velocity_body_frame.copy()
        self.prev_ang_vel = robot.base_angular_velocity_body_frame.copy()


    def _extract_wrench(self, grf: np.ndarray) -> np.ndarray:
        # — your existing wrench extraction unchanged —
        f = grf.reshape(4,3)
        F = np.sum(f, axis=0)
        p = self._qp._robot.foot_positions_in_base_frame[0]
        M = np.zeros(3, dtype=np.float32)
        for i in range(4):
            M += np.cross(p[i], f[i])
        return np.concatenate([F, M], axis=0)


    def _predictor(self, y_meas: np.ndarray):
        # — your existing predictor unchanged —
        inv_mass = self._qp._inv_mass
        inv_I    = self._qp._inv_inertia

        F_ref = self.u_ref[:3]
        M_ref = self.u_ref[3:]
        a_lin = inv_mass.dot(F_ref)
        a_ang = inv_I.dot(M_ref)
        model_acc = np.concatenate([a_lin, a_ang]) + self.d_hat

        xdot = model_acc + self.L.dot(y_meas - self.x_hat)
        self.x_hat += xdot * self.dt


    def _adaptation(self, y_meas: np.ndarray):
        # — your existing adaptation unchanged —
        err = self.x_hat - y_meas
        d_dot = - self.Gamma * err
        self.d_hat += d_dot * self.dt


    def _l1_filter(self) -> np.ndarray:
        # — your existing filter unchanged —
        α = self.omega_c * self.dt / (1 + self.omega_c * self.dt)
        self.filt_buf = α * self.d_hat + (1 - α) * self.filt_buf
        return self.filt_buf


    def get_action(self, foot_contact_state: np.ndarray, swing_foot_position: np.ndarray):
        mc, desired_acc, solved_acc, qp_cost, num_clips = \
            self._qp.get_action(foot_contact_state, swing_foot_position)

        _, _, grf, _, _ = self._qp._solve_joint_torques(foot_contact_state, desired_acc)
        flat_grf = grf.flatten()
        self.log['grf_nominal'].append(flat_grf.copy())
        self.u_ref = self._extract_wrench(flat_grf)

        curr_lin = self._qp._robot.base_velocity_body_frame[0]
        curr_ang = self._qp._robot.base_angular_velocity_body_frame[0]
        meas_lin_acc = (curr_lin - self.prev_lin_vel[0]) / self.dt
        meas_ang_acc = (curr_ang - self.prev_ang_vel[0]) / self.dt
        self.prev_lin_vel[0] = curr_lin.copy()
        self.prev_ang_vel[0] = curr_ang.copy()
        y_meas = np.concatenate([meas_lin_acc, meas_ang_acc])

        self._predictor(y_meas)
        self._adaptation(y_meas)
        delta = self._l1_filter()
        self.log['d_hat'].append(self.d_hat.copy())
        self.log['delta'].append(delta.copy())

        corrected_acc = desired_acc.copy()
        corrected_acc[0, :3]  -= delta[:3]
        corrected_acc[0, 3:]  -= delta[3:]

        _, _, grf_corr, _, _ = \
            self._qp._solve_joint_torques(foot_contact_state, corrected_acc)
        flat_grf_corr = grf_corr.flatten()
        self.log['grf_corrected'].append(flat_grf_corr.copy())
        self.log['time'].append(len(self.log['time']) * self.dt)

        from src.robots.sim2sim.motors import MotorCommand
        jac = self._qp._robot.all_foot_jacobian
        motor_torques_corr = -np.matmul(
            grf_corr[:, None, :],
            jac
        )[0, 0, :]

        mc_corrected = MotorCommand(
            desired_position     = mc.desired_position,
            kp                   = mc.kp,
            desired_velocity     = mc.desired_velocity,
            kd                   = mc.kd,
            desired_extra_torque = motor_torques_corr
        )
        return mc_corrected, corrected_acc, solved_acc, qp_cost, num_clips



    # ---- forwarded properties unchanged ----
    @property
    def desired_base_position(self):           return self._qp.desired_base_position
    @desired_base_position.setter
    def desired_base_position(self, val):      self._qp.desired_base_position = val
    @property
    def desired_base_orientation_rpy(self):    return self._qp.desired_base_orientation_rpy
    @desired_base_orientation_rpy.setter
    def desired_base_orientation_rpy(self, val): self._qp.desired_base_orientation_rpy = val
    @property
    def desired_linear_velocity(self):         return self._qp.desired_linear_velocity
    @desired_linear_velocity.setter
    def desired_linear_velocity(self, val):    self._qp.desired_linear_velocity = val
    @property
    def desired_angular_velocity(self):        return self._qp.desired_angular_velocity
    @desired_angular_velocity.setter
    def desired_angular_velocity(self, val):   self._qp.desired_angular_velocity = val
