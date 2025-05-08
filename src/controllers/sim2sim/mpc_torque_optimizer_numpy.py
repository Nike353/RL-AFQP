import numpy as np
import osqp
from scipy import sparse
from src.controllers.sim2sim.qp_torque_optimizer_numpy import (
    QPTorqueOptimizer,
    construct_mass_mat
)

class L1AdaptiveMPCTorqueOptimizer:
    """
    Replaces the one‑step QP with an N‑step MPC, wrapped in an L1 loop.
    """
    def __init__(self, robot, qp_kwargs: dict, mpc_horizon: int, l1_kwargs: dict):
        import inspect
        sig = inspect.signature(QPTorqueOptimizer.__init__).parameters
        qp_args = {k: v for k, v in qp_kwargs.items() if k in sig}
        self._qp = QPTorqueOptimizer(robot, **qp_args)

        self.N  = mpc_horizon
        self.dt = qp_kwargs['dt']

        # Nominal mass matrix mapping foot GRFs to centroidal acceleration
        foot_contact_full = np.ones((self._qp._robot.num_envs, 4), dtype=bool)
        mass_mat_nom = construct_mass_mat(
            self._qp._robot.foot_positions_in_base_frame,
            foot_contact_full,
            self._qp._inv_mass,
            self._qp._inv_inertia,
            mask_noncontact_legs=not self._qp._use_full_qp
        )[0]  # (6 x 12)
        self.B = mass_mat_nom
        self.A = np.eye(6)

        # L1 gains
        self.L       = l1_kwargs['observer_gain']  # e.g. np.eye(6)*0.1
        self.Gamma   = l1_kwargs['adapt_gain']    # e.g. 0.01 or small scalar
        self.omega_c = l1_kwargs['filter_cutoff']
        self.max_d   = l1_kwargs.get('max_adapt', np.ones(6, dtype=np.float32) * 5.0)
        self.tau_scale = l1_kwargs.get('tau_scale', 0.1)
        self._build_l1_states()
        self._setup_osqp_templates()

    def _build_l1_states(self):
        self.x_hat = np.zeros(6)
        self.d_hat = np.zeros(6)
        self.filt  = np.zeros(6)
        self.u_ref = np.zeros(6)

    def _setup_osqp_templates(self):
        # Hessian for OSQP: block-diagonal Rw per time step
        Rw = sparse.eye(12) * 1e-2
        self.P = sparse.block_diag([Rw]*self.N).tocsc()

        # Friction cone constraints: 4 inequalities per foot per step
        mu = self._qp._foot_friction_coef_adaptive[0] if hasattr(self._qp, '_foot_friction_coef_adaptive') else self._qp._foot_friction_coef
        G_step, h_step = [], []
        for f_idx in range(4):
            for sign in [1, -1]:
                # fx constraint
                vec = np.zeros(12); vec[f_idx*3] = sign; vec[f_idx*3+2] = -mu
                G_step.append(vec.copy()); h_step.append(0.0)
                # fy constraint
                vec = np.zeros(12); vec[f_idx*3+1] = sign; vec[f_idx*3+2] = -mu
                G_step.append(vec.copy()); h_step.append(0.0)
        self.G   = sparse.block_diag([sparse.csc_matrix(np.vstack(G_step))]*self.N).tocsc()
        self.lhs = -np.inf * np.ones(len(h_step) * self.N)
        self.rhs = np.tile(h_step, self.N)

    def _solve_mpc(self, acc_ref, foot_contact_state):
        # Build linear cost term q of length 12N
        q_list = []
        for k in range(self.N):
            err = acc_ref[k] - self.A.dot(self.x_hat)
            mass_mat = construct_mass_mat(
                self._qp._robot.foot_positions_in_base_frame,
                foot_contact_state,
                self._qp._inv_mass,
                self._qp._inv_inertia,
                mask_noncontact_legs=not self._qp._use_full_qp
            )[0]
            qk = mass_mat.T.dot(self._qp._Wq.dot(err))
            q_list.append(qk)
        q = np.hstack(q_list)

        # OSQP constraints
        A_stack = self.G
        l_stack = self.lhs
        u_stack = self.rhs

        prob = osqp.OSQP()
        prob.setup(P=self.P, q=q, A=A_stack, l=l_stack, u=u_stack, verbose=False)
        res = prob.solve()
        return res.x[:12]

    def get_action(self, foot_contact_state, swing_foot_position):
        # 1) Nominal QP solution
        mc, desired_acc, _, _, _ = self._qp.get_action(foot_contact_state, swing_foot_position)
        _, _, grf, _, _ = self._qp._solve_joint_torques(foot_contact_state, desired_acc)
        self.u_ref = self._extract_wrench(grf.flatten())

        # 2) Measure actual centroidal accel via finite diff
        lin = self._qp._robot.base_velocity_body_frame[0]
        ang = self._qp._robot.base_angular_velocity_body_frame[0]
        meas_lin = (lin - getattr(self, 'prev_lin', lin)) / self.dt
        meas_ang = (ang - getattr(self, 'prev_ang', ang)) / self.dt
        self.prev_lin, self.prev_ang = lin.copy(), ang.copy()
        y_meas = np.hstack([meas_lin, meas_ang])

        # 3) L1 predictor-corrector
        self.x_hat += ((self.u_ref + self.d_hat) + self.L.dot(y_meas - self.x_hat)) * self.dt
        err = self.x_hat - y_meas
        d_dot = -self.Gamma.dot(err) if isinstance(self.Gamma, np.ndarray) else -self.Gamma * err
        self.d_hat += d_dot * self.dt
        self.d_hat = np.clip(self.d_hat, -self.max_d, self.max_d)
        alpha = self.omega_c * self.dt / (1 + self.omega_c * self.dt)
        self.filt = alpha * self.d_hat + (1 - alpha) * self.filt

        # 4) Correct reference accel and tile
        corr = desired_acc.copy()
        corr[0, :3] -= self.filt[:3]
        corr[0, 3:] -= self.filt[3:]
        acc_ref = np.tile(corr[0], (self.N, 1))

        # 5) Solve MPC QP for foot forces
        f0 = self._solve_mpc(acc_ref, foot_contact_state)

        # 6) Map foot forces to joint torques
        J = self._qp._robot.all_foot_jacobian[0]  # (12×12)
        tau = -J.T.dot(f0)
        tau = tau * self.tau_scale

        from src.robots.sim2sim.motors import MotorCommand
        return MotorCommand(mc.desired_position, mc.kp, mc.desired_velocity, mc.kd, tau), corr, None, None, None

    def _extract_wrench(self, grf):
        f = grf.reshape(4, 3)
        F = f.sum(axis=0)
        p = self._qp._robot.foot_positions_in_base_frame[0]
        M = sum(np.cross(p[i], f[i]) for i in range(4))
        return np.hstack([F, M])

    # Forward centroidal commands to underlying QP optimizer
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

    @property
    def desired_linear_acceleration(self): return self._qp.desired_linear_acceleration
    @desired_linear_acceleration.setter
    def desired_linear_acceleration(self, val): self._qp.desired_linear_acceleration = val

    @property
    def desired_angular_acceleration(self): return self._qp.desired_angular_acceleration
    @desired_angular_acceleration.setter
    def desired_angular_acceleration(self, val): self._qp.desired_angular_acceleration = val
