"""Solves the centroidal QP to compute desired foot torques."""
import time


import numpy as np
import torch
from qpth.qp import QPFunction, QPSolvers

from src.robots.sim2sim.motors import MotorCommand
from src.utilities.rotation_utils import quat_to_rot_mat_np

def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Performs batch matrix multiplication of two 3D arrays.
    Equivalent to torch.bmm(A, B).
    
    Args:
        A: NumPy array of shape (B, N, M)
        B: NumPy array of shape (B, M, P)
    
    Returns:
        NumPy array of shape (B, N, P)
    """
    return np.matmul(A, B)

def quater_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions q1 and q2.
    Quaternions are in the (w, x, y, z) format.
    
    Args:
        q1: NumPy array of shape (..., 4) in (w, x, y, z) format
        q2: NumPy array of shape (..., 4) in (w, x, y, z) format
    
    Returns:
        A NumPy array of shape (..., 4) representing the quaternion product in (w, x, y, z) format.
    """
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.concatenate([w, x, y, z], axis=-1)

def quater_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotates a vector `v` using a quaternion `q`.

    Args:
        q: NumPy array of shape (..., 4), representing quaternions in (x, y, z, w) format.
        v: NumPy array of shape (..., 3), representing vectors.

    Returns:
        Rotated vector of shape (..., 3).
    """
    # Convert vector to quaternion form (w, x, y, z) with w=0
    v_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)  # (..., 4)

    # Compute conjugate of q: negate vector part (x, y, z), keep scalar part (w)
    q_conj = np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


    # Perform quaternion-vector rotation: v' = q * v * q^(-1)
    v_rotated = quater_mul(quater_mul(q, v_quat), q_conj)

    # Return only the vector part (x, y, z)
    return v_rotated[..., 1:]

def quaternion_to_axis_angle(q):
    """
    Converts quaternions in (w, x, y, z) format to axis-angle representation.

    Args:
        q: NumPy array of shape (N, 4), quaternions in (w, x, y, z)

    Returns:
        axis: (N, 3) normalized axis of rotation
        angle: (N, 1) rotation angle in radians
    """
    # Extract angle from scalar part
    angle = 2 * np.arccos(np.clip(q[:, 0], -0.99999, 0.99999))[:, None]

    # Extract vector part and normalize
    norm = np.clip(np.linalg.norm(q[:, 1:], axis=1), 1e-5, 1)[:, None]
    axis = q[:, 1:] / norm

    return axis, angle


def quater_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (XYZ order) to a quaternion (x, y, z, w).
    
    Args:
        roll: Rotation around X-axis (in radians).
        pitch: Rotation around Y-axis (in radians).
        yaw: Rotation around Z-axis (in radians).

    Returns:
        A NumPy array of shape (..., 4) representing the quaternion in (x, y, z, w) format.
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.stack([w, x, y, z], axis=-1)


def compute_orientation_error(desired_orientation_rpy,
                              base_orientation_quat,
                              device: str = 'cuda'):
    """
    Computes orientation error in base frame, given desired orientation (RPY)
    and current base orientation in quaternion (w, x, y, z) format.

    Args:
        desired_orientation_rpy: (N, 3) Euler angles (roll, pitch, yaw)
        base_orientation_quat: (N, 4) quaternions in (w, x, y, z)
        device: not used here, placeholder for PyTorch compatibility

    Returns:
        (N, 3) orientation error in base frame
    """
    # Construct desired quaternion with yaw = 0
    desired_quat = quater_from_euler_xyz(
        desired_orientation_rpy[:, 0],  # roll
        desired_orientation_rpy[:, 1],  # pitch
        np.zeros_like(desired_orientation_rpy[:, 2])  # yaw = 0
    )

    # Invert base orientation: conjugate = [w, -x, -y, -z]
    base_quat_inv = np.copy(base_orientation_quat)
    base_quat_inv[:, 1:] *= -1

    # Compute error quaternion: q_error = q_desired * q_base_conjugate
    error_quat = quater_mul(desired_quat, base_quat_inv)

    # Convert error quaternion to axis-angle
    axis, angle = quaternion_to_axis_angle(error_quat)

    # Wrap angle to [-pi, pi]
    angle = np.where(angle > np.pi, angle - 2 * np.pi, angle)

    # Compute orientation error in world frame
    error_so3 = axis * angle  # shape: (N, 3)

    # Rotate error into base frame
    return quater_rotate(base_orientation_quat, error_so3)



def compute_desired_acc(
    base_orientation_rpy: np.ndarray,
    base_position: np.ndarray,
    base_angular_velocity_body_frame: np.ndarray,
    base_velocity_body_frame: np.ndarray,
    desired_base_orientation_rpy: np.ndarray,
    desired_base_position: np.ndarray,
    desired_angular_velocity: np.ndarray,
    desired_linear_velocity: np.ndarray,
    desired_angular_acceleration: np.ndarray,
    desired_linear_acceleration: np.ndarray,
    base_position_kp: np.ndarray,
    base_position_kd: np.ndarray,
    base_orientation_kp: np.ndarray,
    base_orientation_kd: np.ndarray,
    device: str = "cuda",
):
  base_rpy = base_orientation_rpy
  base_quat = quater_from_euler_xyz(
      base_rpy[:, 0], base_rpy[:, 1],
      np.zeros_like(base_rpy[:, 0]))
  base_rot_mat = quat_to_rot_mat_np(base_quat)
  base_rot_mat_t = np.transpose(base_rot_mat, (0, 2, 1))


  lin_pos_error = desired_base_position - base_position
  lin_pos_error[:, :2] = 0
  lin_vel_error = desired_linear_velocity - np.matmul(
      base_rot_mat, base_velocity_body_frame[:, :, None])[:, :, 0]
  desired_lin_acc_gravity_frame = (base_position_kp * lin_pos_error +
                                   base_position_kd * lin_vel_error +
                                   desired_linear_acceleration)

  ang_pos_error = compute_orientation_error(desired_base_orientation_rpy,
                                            base_quat)
  ang_vel_error = desired_angular_velocity - np.matmul(
      base_rot_mat, base_angular_velocity_body_frame[:, :, None])[:, :, 0]
  desired_ang_acc_gravity_frame = (base_orientation_kp * ang_pos_error +
                                   base_orientation_kd * ang_vel_error +
                                   desired_angular_acceleration)

  desired_lin_acc_body_frame = np.matmul(
      base_rot_mat_t, desired_lin_acc_gravity_frame[:, :, None])[:, :, 0]
  desired_ang_acc_body_frame = np.matmul(
      base_rot_mat_t, desired_ang_acc_gravity_frame[:, :, None])[:, :, 0]
  # print(f"Desired position: {desired_base_position}")
  # print(f"Current position: {base_position}")
  # print(f"Desired lin acc body: {desired_lin_acc_body_frame}")
  # print(f"Desired ang acc body: {desired_ang_acc_body_frame}")
  # ans = input("Any Key...")
  # if ans in ["y", "Y"]:
  #   import pdb
  #   pdb.set_trace()
  return np.concatenate(
      (desired_lin_acc_body_frame, desired_ang_acc_body_frame), axis=1)


def convert_to_skew_symmetric_batch(foot_positions):
  """
  Converts foot positions (nx4x3) into skew-symmetric ones (nx3x12)
  """
  n = foot_positions.shape[0]
  x = foot_positions[:, :, 0]
  y = foot_positions[:, :, 1]
  z = foot_positions[:, :, 2]
  zero = np.zeros_like(x)
  skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=1).reshape(
      (n, 3, 3, 4))
  return np.concatenate(
      [skew[:, :, :, 0], skew[:, :, :, 1], skew[:, :, :, 2], skew[:, :, :, 3]],
      axis=2)


def construct_mass_mat(foot_positions,
                       foot_contact_state,
                       inv_mass,
                       inv_inertia,
                       device: str = 'cuda',
                       mask_noncontact_legs: bool = True):
  num_envs = foot_positions.shape[0]
  mass_mat = np.zeros((num_envs, 6, 12))
  # Construct mass matrix
  inv_mass_concat = np.concatenate([inv_mass] * 4, axis=2)
  mass_mat[:, :3, :] = inv_mass_concat
  px = convert_to_skew_symmetric_batch(foot_positions)
  mass_mat[:, 3:6] = np.matmul(inv_inertia, px)
  # Mark out non-contact legs
  if mask_noncontact_legs:
    
    non_contact_indices = np.nonzero(np.logical_not(foot_contact_state))
    if len(non_contact_indices[0]) > 0:
      non_contact_indices = np.hstack((non_contact_indices[0][:, np.newaxis], non_contact_indices[1][:, np.newaxis]))
      env_id, leg_id = non_contact_indices[:, 0], non_contact_indices[:, 1]
      mass_mat[env_id, :, leg_id * 3] = 0
      mass_mat[env_id, :, leg_id * 3 + 1] = 0
      mass_mat[env_id, :, leg_id * 3 + 2] = 0
  return mass_mat


def solve_grf(mass_mat,
              desired_acc,
              base_rot_mat_t,
              Wq,
              Wf: float,
              foot_friction_coef: float,
              clip_grf: bool,
              foot_contact_state,
              device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  g = np.zeros((num_envs, 6))
  g[:, 2] = 9.8

  g[:, :3] = np.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]
  Q = np.zeros((num_envs, 6, 6)) + Wq[None, :]
  Wf_mat = np.eye(12) * Wf
  R = np.zeros((num_envs, 12, 12)) + Wf_mat[None, :]

  mass_mat_T = np.transpose(mass_mat, (0, 2, 1))
  quad_term = np.matmul(np.matmul(mass_mat_T, Q), mass_mat) + R
  linear_term = np.matmul(np.matmul(mass_mat_T, Q), (g + desired_acc)[:, :, None])[:, :, 0]

  grf = np.linalg.solve(quad_term, linear_term)

  base_rot_mat = np.transpose(base_rot_mat_t, (0,1, 2))
  grf = grf.reshape((-1, 4, 3))
  grf_world = np.transpose(
      np.matmul(base_rot_mat, np.transpose(grf, (0, 2, 1))), (0, 2, 1))

  if clip_grf:
    grf_world[:, :, 2] = grf_world[:, :, 2].clip(min=10, max=130)
    grf_world[:, :, 2] *= foot_contact_state
  friction_force = np.linalg.norm(grf_world[:, :, :2], axis=2) + 0.001
  max_friction_force = foot_friction_coef * grf_world[:, :, 2].clip(min=0)
  multiplier = np.where(friction_force < max_friction_force, 1,
                           max_friction_force / friction_force)
  if clip_grf:
    grf_world[:, :, :2] *= multiplier[:, :, None]
  grf = np.transpose(
      np.matmul(base_rot_mat_t, np.transpose(grf_world, (0, 2, 1))), (0, 2, 1))

  grf = grf.reshape((-1, 12))

  # Convert to motor torques
  solved_acc = np.matmul(mass_mat, grf[:, :, None])[:, :, 0] - g
  qp_cost = np.matmul(
      np.matmul((solved_acc - desired_acc)[:, :, np.newaxis].transpose(0, 2, 1), Q),
      (solved_acc - desired_acc)[:, :, np.newaxis])[:, 0, 0]

  return grf, solved_acc, qp_cost, np.sum(
      friction_force > max_friction_force + 1, axis=1)


def solve_grf_qpth(mass_mat,
                   desired_acc,
                   base_rot_mat_t,
                   Wq,
                   Wf: float,
                   foot_friction_coef: float,
                   clip_grf: bool,
                   foot_contact_state,
                   device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  g = np.zeros((num_envs, 6))
  g[:, 2] = 9.8

  g[:, :3] = np.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]
  Q = np.zeros((num_envs, 6, 6)) + Wq[None, :]
  Wf_mat = np.eye(12) * Wf
  R = np.zeros((num_envs, 12, 12)) + Wf_mat[None, :]
  mass_mat_T = np.transpose(mass_mat, (0, 2, 1))  # Transpose along axis (1, 2)
  quad_term = np.matmul(np.matmul(mass_mat_T, Q), mass_mat) + R
  linear_term = np.matmul(np.matmul(mass_mat_T, Q), (g + desired_acc)[:, :, None])[:, :, 0]

  G = np.zeros((mass_mat.shape[0], 24, 12))
  h = np.zeros((mass_mat.shape[0], 24)) + 1e-3
  base_rot_mat = np.transpose(base_rot_mat_t, (0,1, 2))
  for leg_id in range(4):
    G[:, leg_id * 2, leg_id * 3 + 2] = 1
    G[:, leg_id * 2 + 1, leg_id * 3 + 2] = -1

    row_id, col_id = 8 + leg_id * 4, leg_id * 3
    G[:, row_id, col_id] = 1
    G[:, row_id, col_id + 2] = -foot_friction_coef

    G[:, row_id + 1, col_id] = -1
    G[:, row_id + 1, col_id + 2] = -foot_friction_coef

    G[:, row_id + 2, col_id + 1] = 1
    G[:, row_id + 2, col_id + 2] = -foot_friction_coef

    G[:, row_id + 3, col_id + 1] = -1
    G[:, row_id + 3, col_id + 2] = -foot_friction_coef
    G[:, row_id:row_id + 4, col_id:col_id + 3] = np.matmul(
        G[:, row_id:row_id + 4, col_id:col_id + 3], base_rot_mat)

  contact_ids = foot_contact_state.nonzero()

  h[contact_ids[:, 0], contact_ids[:, 1] * 2] = 130
  h[contact_ids[:, 0], contact_ids[:, 1] * 2 + 1] = -10
  e = np.array([])

  qf = QPFunction(verbose=-1,
                  check_Q_spd=False,
                  eps=1e-3,
                  solver=QPSolvers.PDIPM_BATCHED)
  grf = qf(quad_term.double(), -linear_term.double(), G.double(), h.double(),
           e, e).float()
  # print(grf)
  # ans = input("Any Key...")
  # if ans in ["Y", "y"]:
  #   import pdb
  #   pdb.set_trace()
  solved_acc = np.matmul(mass_mat, grf[:, :, None])[:, :, 0] - g
  qp_cost = np.matmul(
      np.matmul((solved_acc - desired_acc)[:, :, np.newaxis].transpose(0, 2, 1), Q),
      (solved_acc - desired_acc)[:, :, np.newaxis])[:, 0, 0]


  return grf, solved_acc, qp_cost, np.zeros(mass_mat.shape[0])


class QPTorqueOptimizer:
  """Centroidal QP controller to optimize for joint torques."""
  def __init__(self,
               robot,
               base_position_kp=np.array([0., 0., 50]),
               base_position_kd=np.array([10., 10., 10.]),
               base_orientation_kp=np.array([50., 50., 0.]),
               base_orientation_kd=np.array([10., 10., 10.]),
               weight_ddq=np.diag([1., 1., 10., 10., 10., 1.]),
               weight_grf=1e-4,
               body_mass=13.076,
               body_inertia=np.array([0.14, 0.35, 0.35]) * 0.5,
               desired_body_height=0.26,
               foot_friction_coef=0.7,
               clip_grf=False,
               use_full_qp=False,
               adaptive_gains=None,
               dt=0.002):
    """Initializes the controller with desired weights and gains."""
    self._robot = robot
    self._num_envs = self._robot.num_envs
    self._clip_grf = clip_grf
    self._use_full_qp = use_full_qp
    self._dt = dt

    self._base_orientation_kp = base_orientation_kp
    self._base_orientation_kp = np.stack([self._base_orientation_kp] *
                                            self._num_envs,
                                            axis=0)
    self._base_orientation_kd = base_orientation_kd
    self._base_orientation_kd = np.stack([self._base_orientation_kd] *
                                            self._num_envs,
                                            axis=0)
    self._base_position_kp = base_position_kp
    self._base_position_kp = np.stack([self._base_position_kp] *
                                         self._num_envs,
                                         axis=0)
    self._base_position_kd = base_position_kd
    self._base_position_kd = np.stack([self._base_position_kd] *
                                         self._num_envs,
                                         axis=0)
    self._desired_base_orientation_rpy = np.zeros((self._num_envs, 3))
    self._desired_base_position = np.zeros((self._num_envs, 3))
    self._desired_base_position[:, 2] = desired_body_height
    self._desired_linear_velocity = np.zeros((self._num_envs, 3))
    self._desired_angular_velocity = np.zeros((self._num_envs, 3))
    self._desired_linear_acceleration = np.zeros((self._num_envs, 3))
    self._desired_angular_acceleration = np.zeros((self._num_envs, 3))
    self._Wq = np.array(weight_ddq, dtype=np.float32)
    self._Wf = np.array(weight_grf, dtype=np.float32)
    self._foot_friction_coef = foot_friction_coef
    self._inv_mass = np.eye(3) / body_mass
    self._inv_inertia = np.linalg.inv(
        np.diag(np.array(body_inertia, dtype=np.float32)))
    
     # --- Adaptive Parameters ---
    self.mass_hat = np.full((self._num_envs,), np.array(body_mass, dtype=np.float32))
    # Assuming inertia_hat stores diagonal elements for simplicity, or full 3x3 if needed
    # For sim2sim (num_envs=1), this will be a single 3x3 matrix or a (3,) vector.
    self.inertia_hat = np.stack([np.diag(np.array(body_inertia, dtype=np.float32))] * self._num_envs, axis=0)
    self.friction_coef_hat = np.full((self._num_envs,), np.array(foot_friction_coef, dtype=np.float32))

    # Store these as they will be updated by adaptive logic
    self._inv_mass_adaptive = np.stack([np.eye(3, dtype=np.float32) / self.mass_hat[i] for i in range(self._num_envs)], axis=0)
    self._inv_inertia_adaptive = np.stack([np.linalg.inv(self.inertia_hat[i]) for i in range(self._num_envs)], axis=0)
    self._foot_friction_coef_adaptive = self.friction_coef_hat.copy()


    default_gains = {'gamma_mass': 0.1, 'gamma_inertia_diag': 0.05, 'gamma_friction': 0.2}
    if adaptive_gains is None:
        adaptive_gains = default_gains

    self.gamma_mass = np.array(adaptive_gains.get('gamma_mass', default_gains['gamma_mass']), dtype=np.float32)
    self.gamma_inertia_diag = np.array(adaptive_gains.get('gamma_inertia_diag', default_gains['gamma_inertia_diag']), dtype=np.float32) # For diagonal elements
    self.gamma_friction = np.array(adaptive_gains.get('gamma_friction', default_gains['gamma_friction']), dtype=np.float32)

    # For calculating measured acceleration via finite differences
    self.prev_base_lin_vel_body = np.zeros((self._num_envs, 3), dtype=np.float32)
    self.prev_base_ang_vel_body = np.zeros((self._num_envs, 3), dtype=np.float32)

  def _update_estimates(self, desired_acc_body_frame, solved_acc_body_frame, grf_body_frame, foot_contact_state):
      # For num_envs = 1 (typical for sim2sim numpy)
      env_idx = 0 # Assuming single environment for numpy version

      # 1. Calculate/Measure actual base acceleration (world or body frame)
      current_lin_vel_body = self._robot.base_velocity_body_frame[env_idx]
      current_ang_vel_body = self._robot.base_angular_velocity_body_frame[env_idx]

      # Measured acceleration (simple finite difference, consider filtering for robustness)
      measured_lin_acc_body = (current_lin_vel_body - self.prev_base_lin_vel_body[env_idx]) / self._dt
      measured_ang_acc_body = (current_ang_vel_body - self.prev_base_ang_vel_body[env_idx]) / self._dt

      self.prev_base_lin_vel_body[env_idx] = current_lin_vel_body.copy()
      self.prev_base_ang_vel_body[env_idx] = current_ang_vel_body.copy()

      # Error signal for mass/inertia: e_acc = measured_acc - solved_acc
      # solved_acc_body_frame is the acceleration the QP's model predicted with its GRFs.
      e_lin_acc = measured_lin_acc_body - solved_acc_body_frame[env_idx, :3]
      e_ang_acc = measured_ang_acc_body - solved_acc_body_frame[env_idx, 3:]

      # --- Mass Adaptation (force-balance update) ---
      # Total vertical GRF from all feet (in body frame)
      grf = grf_body_frame[env_idx].reshape((4, 3))
      f_tot_z    = np.sum(grf[:, 2])
      # Measured vertical accel + gravity
      meas_acc_z = measured_lin_acc_body[2]
      g_z_body   = self._robot.projected_gravity[env_idx, 2] * 9.81
      # Force error drives mass update:  f_tot_z ≈ m_hat*(meas_acc_z+g_z)
      error_force_z = f_tot_z - self.mass_hat[env_idx] * (meas_acc_z + g_z_body)
      self.mass_hat[env_idx] += self.gamma_mass * error_force_z * self._dt
      # clamp to physical bounds
      self.mass_hat[env_idx] = np.clip(self.mass_hat[env_idx], 5.0, 25.0)

      # --- Inertia Adaptation (diagonal elements for simplicity) ---
      current_I_diag = np.diag(self.inertia_hat[env_idx]).copy()
      delta_Ixx = -self.gamma_inertia_diag * e_ang_acc[0] * solved_acc_body_frame[env_idx, 3]
      delta_Iyy = -self.gamma_inertia_diag * e_ang_acc[1] * solved_acc_body_frame[env_idx, 4]
      delta_Izz = -self.gamma_inertia_diag * e_ang_acc[2] * solved_acc_body_frame[env_idx, 5]

      current_I_diag[0] = np.clip(current_I_diag[0] + delta_Ixx * self._dt, 0.01, 0.5)
      current_I_diag[1] = np.clip(current_I_diag[1] + delta_Iyy * self._dt, 0.01, 1.0)
      current_I_diag[2] = np.clip(current_I_diag[2] + delta_Izz * self._dt, 0.01, 1.0)
      self.inertia_hat[env_idx] = np.diag(current_I_diag)

      # --- Friction Adaptation ---
      grf_feet_body = grf_body_frame[env_idx].reshape((4, 3))
      # Transform GRFs to world frame for friction calculation
      base_rot_mat_env = self._robot.base_rot_mat[env_idx]
      grf_feet_world = np.dot(base_rot_mat_env, grf_feet_body.T).T

      foot_velocities_world = self._robot.foot_velocities_in_world_frame[env_idx]

      for leg_idx in range(4):
          if foot_contact_state[env_idx, leg_idx]: # If foot is in contact
              N_i = grf_feet_world[leg_idx, 2] # Normal force
              if N_i > 1.0: # adapt only if there's significant normal force
                  F_tangential_mag_i = np.linalg.norm(grf_feet_world[leg_idx, :2])
                  
                  foot_speed_sq = np.sum(np.square(foot_velocities_world[leg_idx, :2]))
                  IS_SLIPPING_THRESHOLD_SQ = 0.01**2 # (m/s)^2, heuristic
                  
                  if foot_speed_sq > IS_SLIPPING_THRESHOLD_SQ:
                      # If slipping, tangential force magnitude should be mu_hat * N
                      current_mu_at_contact = F_tangential_mag_i / (N_i + 1e-6) # Add epsilon for stability
                      delta_friction = self.gamma_friction * (current_mu_at_contact - self.friction_coef_hat[env_idx])
                      self.friction_coef_hat[env_idx] = np.clip(self.friction_coef_hat[env_idx] + delta_friction * self._dt, 0.1, 1.5)

      # Update the cached adaptive parameters for the QP
      self._inv_mass_adaptive[env_idx] = np.eye(3, dtype=np.float32) / self.mass_hat[env_idx]
      self._inv_inertia_adaptive[env_idx] = np.linalg.inv(self.inertia_hat[env_idx])
      self._foot_friction_coef_adaptive[env_idx] = self.friction_coef_hat[env_idx]


  def _solve_joint_torques_adaptive(self, foot_contact_state, desired_com_ddq):
    # Uses self._inv_mass_adaptive, self._inv_inertia_adaptive, self._foot_friction_coef_adaptive
    # which are updated by _update_estimates

    mass_mat_adaptive = construct_mass_mat( # construct_mass_mat needs to take inv_mass and inv_inertia
        self._robot.foot_positions_in_base_frame, # Shape (num_envs, 4, 3)
        foot_contact_state,                       # Shape (num_envs, 4)
        self._inv_mass_adaptive,                  # Shape (num_envs, 3, 3)
        self._inv_inertia_adaptive,               # Shape (num_envs, 3, 3)
        mask_noncontact_legs=not self._use_full_qp
    )

    # Ensure friction coefficient is correctly shaped for solve_grf/solve_grf_qpth
    # If solve_grf expects a scalar and num_envs is 1, pass self._foot_friction_coef_adaptive[0]
    friction_coeff_for_solver = self._foot_friction_coef_adaptive[0] if self._num_envs == 1 else self._foot_friction_coef_adaptive

    if self._use_full_qp:
        # solve_grf_qpth needs to be compatible with numpy and potentially per-env friction_coef
        grf, solved_acc, qp_cost, num_clips = solve_grf_qpth(
            mass_mat_adaptive,
            desired_com_ddq,
            self._robot.base_rot_mat_t,
            self._Wq,
            self._Wf,
            friction_coeff_for_solver, 
            self._clip_grf,
            foot_contact_state
        )
    else:
        # solve_grf needs to be compatible with numpy and potentially per-env friction_coef
        grf, solved_acc, qp_cost, num_clips = solve_grf(
            mass_mat_adaptive,
            desired_com_ddq,
            self._robot.base_rot_mat_t,
            self._Wq,
            self._Wf,
            friction_coeff_for_solver,
            self._clip_grf,
            foot_contact_state
        )

    all_foot_jacobian = self._robot.all_foot_jacobian # Shape (num_envs, 12, 12)
    # grf is (num_envs, 12)
    motor_torques = -np.matmul(grf[:, np.newaxis, :], all_foot_jacobian)[:, 0, :]
    return motor_torques, solved_acc, grf, qp_cost, num_clips

  def compute_joint_command(self, foot_contact_state: np.ndarray,
                            desired_base_orientation_rpy: np.ndarray,
                            desired_base_position: np.ndarray,
                            desired_foot_position: np.ndarray,
                            desired_angular_velocity: np.ndarray,
                            desired_linear_velocity: np.ndarray,
                            desired_foot_velocity: np.ndarray,
                            desired_angular_acceleration: np.ndarray,
                            desired_linear_acceleration: np.ndarray,
                            desired_foot_acceleration: np.ndarray):
    desired_acc_body_frame = compute_desired_acc(
        self._robot.base_orientation_rpy,
        self._robot.base_position,
        self._robot.base_angular_velocity_body_frame,
        self._robot.base_velocity_body_frame,
        desired_base_orientation_rpy,
        desired_base_position,
        desired_angular_velocity,
        desired_linear_velocity,
        desired_angular_acceleration,
        desired_linear_acceleration,
        self._base_position_kp,
        self._base_position_kd,
        self._base_orientation_kp,
        self._base_orientation_kd)
    desired_acc_body_frame = np.clip(
        desired_acc_body_frame,
        np.array([-30, -30, -10, -20, -20, -20],dtype=np.float32),
        np.array([30, 30, 30, 20, 20, 20], dtype=np.float32))
    motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques(
        foot_contact_state, desired_acc_body_frame)
    foot_position_local = np.matmul(self._robot.base_rot_mat_t,
                                    desired_foot_position.transpose(0, 2, 1)).transpose(0, 2, 1)

    foot_position_local[:, :, 2] = np.clip(foot_position_local[:, :, 2],
                                              -0.35,
                                              -0.1)

    desired_motor_position = self._robot.get_motor_angles_from_foot_positions(
        foot_position_local)

    contact_state_expanded = np.repeat(foot_contact_state, 3, axis=1)


    desired_position = np.where(contact_state_expanded,
                                   self._robot.motor_positions,
                                   desired_motor_position)
    desired_velocity = np.where(contact_state_expanded,
                                   self._robot.motor_velocities,
                                   np.zeros_like(motor_torques))
    desired_torque = np.where(contact_state_expanded, motor_torques,
                                 np.zeros_like(motor_torques))
    desired_torque = np.clip(desired_torque,
                                a_max=self._robot.motor_group.max_torques,
                                a_min=self._robot.motor_group.min_torques)
    # print(self._robot.time_since_reset)
    # print("Contact: {}".format(foot_contact_state))
    # print("Desired pos: {}".format(desired_base_position))
    # print("Current vel: {}".format(self._robot.base_velocity_body_frame))
    # print("Desired vel: {}".format(desired_linear_velocity))
    # print(f"GRF: {grf.reshape((4, 3))}")
    # print("Desired acc: {}".format(desired_acc_body_frame))
    # print("Solved acc: {}".format(solved_acc))
    # print(f"Desired torque: {desired_torque}")
    # ans = input("Any Key...")
    # if ans in ['y', 'Y']:
    #   import pdb
    #   pdb.set_trace()
    return MotorCommand(
        desired_position=desired_position,
        kp=np.ones_like(self._robot.motor_group.kps) * 30,
        desired_velocity=desired_velocity,
        kd=np.ones_like(self._robot.motor_group.kds) * 1,
        desired_extra_torque=desired_torque
    ), desired_acc_body_frame, solved_acc, qp_cost, num_clips

  def get_action(self, foot_contact_state: np.ndarray, swing_foot_position: np.ndarray):
    # 1. Compute desired base accelerations (PD part for base)
    # This part remains largely the same, using current robot state and desired base poses/velocities
    desired_acc_body_frame = compute_desired_acc(
        self._robot.base_orientation_rpy,
        self._robot.base_position,
        self._robot.base_angular_velocity_body_frame,
        self._robot.base_velocity_body_frame,
        self.desired_base_orientation_rpy, # Property uses self._desired_base_orientation_rpy
        self.desired_base_position,    # Property
        self.desired_angular_velocity, # Property
        self.desired_linear_velocity,  # Property
        self._desired_angular_acceleration, # Direct attribute
        self._desired_linear_acceleration,  # Direct attribute
        self._base_position_kp,
        self._base_position_kd,
        self._base_orientation_kp,
        self._base_orientation_kd,
    )
    desired_acc_body_frame = np.clip(
        desired_acc_body_frame,
        np.array([-30, -30, -10, -20, -20, -20], dtype=np.float32),
        np.array([30, 30, 30, 20, 20, 20], dtype=np.float32)
    )

    # 2. Solve QP using current adaptive estimates
    # _solve_joint_torques_adaptive will use self._inv_mass_adaptive etc.
    motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques_adaptive(
        foot_contact_state,
        desired_acc_body_frame
    )

    # 3. Update estimates based on the outcome
    # solved_acc is from the QP model, grf is also from QP (in body frame)
    self._update_estimates(desired_acc_body_frame, solved_acc, grf, foot_contact_state)

    # 4. Compute final motor command (IK for swing legs, torques for stance legs)
    # This logic is similar to the original get_action
    foot_position_local = np.matmul(self._robot.base_rot_mat_t,
                                swing_foot_position.transpose(0, 2, 1)).transpose(0, 2, 1)
    foot_position_local[:, :, 2] = np.clip(foot_position_local[:, :, 2], -0.35, -0.1)

    desired_motor_position = self._robot.get_motor_angles_from_foot_positions(foot_position_local)

    contact_state_expanded = np.repeat(foot_contact_state, 3, axis=1)

    final_desired_position = np.where(contact_state_expanded, self._robot.motor_positions, desired_motor_position)
    final_desired_velocity = np.where(contact_state_expanded, self._robot.motor_velocities, np.zeros_like(motor_torques))
    # Stance legs get QP torque, swing legs get PD torque (effectively, as desired_extra_torque)
    # For swing legs, motor_torques might be zero if not in contact_state_expanded,
    # so PD for position tracking will be dominant.
    final_desired_torque = np.where(contact_state_expanded, motor_torques, np.zeros_like(motor_torques))
    final_desired_torque = np.clip(final_desired_torque,
                                    self._robot.motor_group.min_torques,
                                    self._robot.motor_group.max_torques)

    # Construct MotorCommand
    # The kp and kd here are for joint-level PD control.
    # The QP torques act as feedforward for stance legs.
    # Swing legs will primarily follow desired_position via these Kp/Kd.
    kp_joint = np.ones_like(self._robot.motor_group.kps) * 50 # Example Kp for joint tracking
    kd_joint = np.ones_like(self._robot.motor_group.kds) * 1.0  # Example Kd for joint tracking

    # For swing legs, desired_extra_torque is zero, PD terms will generate torque.
    # For stance legs, desired_position/velocity match current, so PD terms are small,
    # and desired_extra_torque (from QP) dominates.
    motor_command_obj = MotorCommand(
        desired_position=final_desired_position,
        kp=kp_joint, 
        desired_velocity=final_desired_velocity,
        kd=kd_joint,
        desired_extra_torque=final_desired_torque
    )

    return motor_command_obj, desired_acc_body_frame, solved_acc, qp_cost, num_clips

  # Ensure all property setters handle numpy arrays correctly if their types were changed
  @property
  def desired_base_position(self) -> np.ndarray:
      return self._desired_base_position

  @desired_base_position.setter
  def desired_base_position(self, base_position: np.ndarray):
      self._desired_base_position = np.array(base_position, dtype=np.float32)

  @property
  def desired_base_orientation_rpy(self) -> np.ndarray:
      return self._desired_base_orientation_rpy

  @desired_base_orientation_rpy.setter
  def desired_base_orientation_rpy(self, orientation_rpy: np.ndarray):
      self._desired_base_orientation_rpy = np.array(orientation_rpy, dtype=np.float32)

  @property
  def desired_linear_velocity(self) -> np.ndarray:
      return self._desired_linear_velocity

  @desired_linear_velocity.setter
  def desired_linear_velocity(self, desired_linear_velocity: np.ndarray):
      self._desired_linear_velocity = np.array(desired_linear_velocity, dtype=np.float32)

  @property
  def desired_angular_velocity(self) -> np.ndarray:
      return self._desired_angular_velocity

  @desired_angular_velocity.setter
  def desired_angular_velocity(self, desired_angular_velocity: np.ndarray):
      self._desired_angular_velocity = np.array(desired_angular_velocity, dtype=np.float32)

  @property
  def desired_linear_acceleration(self):
      return self._desired_linear_acceleration

  @desired_linear_acceleration.setter
  def desired_linear_acceleration(self, desired_linear_acceleration: np.ndarray):
      self._desired_linear_acceleration = np.array(desired_linear_acceleration, dtype=np.float32)

  @property
  def desired_angular_acceleration(self):
      return self._desired_angular_acceleration

  @desired_angular_acceleration.setter
  def desired_angular_acceleration(self, desired_angular_acceleration: np.ndarray):
      self._desired_angular_acceleration = np.array(desired_angular_acceleration, dtype=np.float32)

