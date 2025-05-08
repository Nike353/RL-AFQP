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
  inv_mass_concat = np.concatenate([inv_mass] * 4, axis=1)
  mass_mat[:, :3] = inv_mass_concat[None, :, :]
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
               dt=0.001):
    """Initializes the controller with desired weights and gains."""
    self._robot = robot
    self._num_envs = self._robot.num_envs
    self._clip_grf = clip_grf
    self._use_full_qp = use_full_qp

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

  def _solve_joint_torques(self, foot_contact_state, desired_com_ddq):
    """Solves centroidal QP to find desired joint torques."""
    self._mass_mat = construct_mass_mat(
        self._robot.foot_positions_in_base_frame,
        foot_contact_state,
        self._inv_mass,
        self._inv_inertia,
        mask_noncontact_legs=not self._use_full_qp)

    # Solve QP
    if self._use_full_qp:
      grf, solved_acc, qp_cost, num_clips = solve_grf_qpth(
          self._mass_mat,
          desired_com_ddq,
          self._robot.base_rot_mat_t,
          self._Wq,
          self._Wf,
          self._foot_friction_coef,
          self._clip_grf,
          foot_contact_state)
    else:
      grf, solved_acc, qp_cost, num_clips = solve_grf(
          self._mass_mat,
          desired_com_ddq,
          self._robot.base_rot_mat_t,
          self._Wq,
          self._Wf,
          self._foot_friction_coef,
          self._clip_grf,
          foot_contact_state)

    all_foot_jacobian = self._robot.all_foot_jacobian
    motor_torques = -np.matmul(grf[:, None, :], all_foot_jacobian)[:, 0]
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

  def get_action(self, foot_contact_state: np.ndarray,
                 swing_foot_position: np.ndarray):
    """Computes motor actions."""
    return self.compute_joint_command(
        foot_contact_state=foot_contact_state,
        desired_base_orientation_rpy=self._desired_base_orientation_rpy,
        desired_base_position=self._desired_base_position,
        desired_foot_position=swing_foot_position,
        desired_angular_velocity=self._desired_angular_velocity,
        desired_linear_velocity=self._desired_linear_velocity,
        desired_foot_velocity=np.zeros(12),
        desired_angular_acceleration=self._desired_angular_acceleration,
        desired_linear_acceleration=self._desired_linear_acceleration,
        desired_foot_acceleration=np.zeros(12))

  def get_action_with_acc(
      self,
      foot_contact_state: np.ndarray,
      desired_acc_body_frame: np.ndarray,
      desired_foot_position: np.ndarray,
  ):
    motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques(
        foot_contact_state, desired_acc_body_frame)
    foot_position_local = np.matmul(self._robot.base_rot_mat_t,
                                    desired_foot_position.transpose(0, 2, 1)).transpose(0, 2, 1)

    foot_position_local[:, :, 2] = np.clip(foot_position_local[:, :, 2],
                                              min=-0.35,
                                              max=-0.1)

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
                                max=self._robot.motor_group.max_torques,
                                min=self._robot.motor_group.min_torques)
    return MotorCommand(
        desired_position=desired_position,
        kp=np.ones_like(self._robot.motor_group.kps) * 30,
        desired_velocity=desired_velocity,
        kd=np.ones_like(self._robot.motor_group.kds) * 1,
        desired_extra_torque=desired_torque
    ), desired_acc_body_frame, solved_acc, qp_cost, num_clips

  @property
  def desired_base_position(self) -> np.ndarray:
    return self._desired_base_position

  @desired_base_position.setter
  def desired_base_position(self, base_position: float):
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
    self._desired_linear_velocity = np.array(desired_linear_velocity)

  @property
  def desired_angular_velocity(self) -> np.ndarray:
    return self._desired_angular_velocity

  @desired_angular_velocity.setter
  def desired_angular_velocity(self, desired_angular_velocity: np.ndarray):
    self._desired_angular_velocity = np.array(desired_angular_velocity)

  @property
  def desired_linear_acceleration(self):
    return self._desired_linear_acceleration

  @desired_linear_acceleration.setter
  def desired_linear_acceleration(self,
                                  desired_linear_acceleration: np.ndarray):
    self._desired_linear_acceleration = np.array(desired_linear_acceleration)

  @property
  def desired_angular_acceleration(self):
    return self._desired_angular_acceleration

  @desired_angular_acceleration.setter
  def desired_angular_acceleration(self,
                                   desired_angular_acceleration: np.ndarray):
    self._desired_angular_acceleration = np.array(desired_angular_acceleration)
