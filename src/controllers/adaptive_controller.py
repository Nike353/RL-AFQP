import time
import numpy as np
import pinocchio as pin

from src.robots.sim2sim.motors import MotorCommand
from src.utilities.rotation_utils import quat_to_rot_mat_np

class AdaptiveController:
    """
    NumPy-based Adaptive Controller for quadruped robots using MRAC.

    Adapts mass/inertia parameters online using joint-only Pinocchio model.
    """
    def __init__(self,
                 robot,            # Go1 wrapper with motor state
                 kp,               # Joint Kp gains (12,)
                 kd,               # Joint Kd gains (12,)
                 gamma,            # Adaptation gain(s), scalar or vector
                 lambda_param,     # Sliding surface gain(s), scalar or vector
                 num_robot_params, # Number of adaptive parameters
                 dt):              # Control timestep
        self._robot = robot
        self._dt = dt
        self._num_params = num_robot_params

        urdf_path = "data/go1/urdf/go1.urdf"
        try:
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
        except Exception as e:
            raise ValueError(f"Cannot load URDF {urdf_path}: {e}")
        self.pin_data = self.pin_model.createData()
        self.nq = self.pin_model.nq  # joints = 12
        self.nv = self.pin_model.nv  # velocities = 12

        kp_arr = np.asarray(kp, dtype=np.float64)
        kd_arr = np.asarray(kd, dtype=np.float64)
        assert kp_arr.size == self.nv
        assert kd_arr.size == self.nv
        self.kp_mat = np.diag(kp_arr)
        self.kd_mat = np.diag(kd_arr)

        gamma_arr = np.asarray(gamma, dtype=np.float64)
        if gamma_arr.size == 1:
            self.gamma_mat = np.eye(self._num_params) * gamma_arr.item()
        elif gamma_arr.size == self._num_params:
            self.gamma_mat = np.diag(gamma_arr)
        else:
            raise ValueError("gamma must be scalar or length num_robot_params")

        lam_arr = np.asarray(lambda_param, dtype=np.float64)
        if lam_arr.size == 1:
            self.lambda_mat = np.eye(self.nv) * lam_arr.item()
        elif lam_arr.size == self.nv:
            self.lambda_mat = np.diag(lam_arr)
        else:
            raise ValueError("lambda_param must be scalar or length nv")

        self.theta_hat = np.zeros((self._num_params,1), dtype=np.float64)

    def _update_pinocchio_state(self):
        qj = self._robot.motor_positions[0]     # (12,)
        vj = self._robot.motor_velocities[0]    # (12,)
        pin.crba(self.pin_model, self.pin_data, qj)
        pin.nonLinearEffects(self.pin_model, self.pin_data, qj, vj)
        return qj, vj

    def _compute_regressor(self, qj, vj, qdd_ref):
        # mass matrix and nonlinear effects
        M   = self.pin_data.M       # (12×12)
        nle = self.pin_data.nle     # (12,)
        ref = M.dot(qdd_ref) + nle  # (12,)
        if self._num_params == 1:
            return ref.reshape(-1,1)
        Y = np.zeros((self.nv,self._num_params), dtype=np.float64)
        Y[:,0] = M.dot(qdd_ref)
        if self._num_params>1:
            Y[:,1] = nle
        return Y

    def _update_parameters(self, Y, s_j):
        # adaptation law: dot(theta) = -Gamma * Y^T * s
        s_col = s_j.reshape(-1,1)
        dtheta = - self.gamma_mat.dot(Y.T).dot(s_col)
        self.theta_hat += dtheta * self._dt

    def compute_torque(self, q_des, qd_des, qdd_des):
        B= q_des.shape[0]
        out = []
        for i in range(B):
            qj, vj = self._update_pinocchio_state()   # (12,)
            e   = q_des[i] - qj
            ed  = qd_des[i] - vj
            s_j = ed + self.lambda_mat.dot(e)
            Y   = self._compute_regressor(qj, vj, qdd_des[i])  # (12×p)
            tau_pd = self.kp_mat.dot(e) + self.kd_mat.dot(ed)
            tau_ad = Y.dot(self.theta_hat).flatten()
            self._update_parameters(Y, s_j)
            out.append(tau_pd + tau_ad)
        return np.vstack(out)

    def get_action(self, q_des, qd_des, qdd_des):
        qj = q_des[:,6:].cpu().numpy()
        vj = qd_des[:,6:].cpu().numpy()
        aj = qdd_des[:,6:].cpu().numpy()
       
        tau = self.compute_torque(qj,vj,aj)  # (B×12)

        B = tau.shape[0]
        kp_vec = np.diag(self.kp_mat)
        kd_vec = np.diag(self.kd_mat)
        kp_np  = np.tile(kp_vec,(B,1))
        kd_np  = np.tile(kd_vec,(B,1))
        return MotorCommand(
            desired_position     = qj,
            kp                   = kp_np,
            desired_velocity     = vj,
            kd                   = kd_np,
            desired_extra_torque = tau
        )
