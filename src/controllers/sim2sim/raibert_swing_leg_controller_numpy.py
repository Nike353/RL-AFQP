"""Implements the Raibert Swing Leg controller in Isaac."""

from typing import Any



import numpy as np


def cubic_bezier(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    progress = np.power(t, 3) + 3 * np.power(t, 2) * (1 - t)
    return x0 + progress * (x1 - x0)


def _gen_swing_foot_trajectory(input_phase: np.ndarray,
                               start_pos: np.ndarray, mid_pos: np.ndarray,
                               end_pos: np.ndarray) -> np.ndarray:
  cutoff = 0.5
  input_phase = np.stack([input_phase] * 3, axis=-1)
  return np.where(
      input_phase < cutoff,
      cubic_bezier(start_pos, mid_pos, input_phase / cutoff),
      cubic_bezier(mid_pos, end_pos, (input_phase - cutoff) / (1 - cutoff)))


def cross_quad(v1, v2):
  """Assumes v1 is nx3, v2 is nx4x3"""
  v1 = np.stack([v1, v1, v1, v1], axis=1)
  shape = v1.shape
  v1 = v1.reshape((-1, 3))
  v2 = v2.reshape((-1, 3))
  return np.cross(v1, v2).reshape((shape[0], shape[1], 3))


def compute_desired_foot_positions(
    base_rot_mat,
    base_height,
    hip_positions_in_body_frame,
    base_velocity_world_frame,
    base_angular_velocity_body_frame,
    projected_gravity,
    desired_base_height: float,
    foot_height: float,
    foot_landing_clearance: float,
    stance_duration,
    normalized_phase,
    phase_switch_foot_positions,
):
  hip_position = np.matmul(base_rot_mat, hip_positions_in_body_frame.transpose(0, 2, 1)).transpose(0, 2, 1)

  # Mid-air position
  mid_position = np.copy(hip_position)
  mid_position[..., 2] = (-base_height[:, None] + foot_height)
  #/ projected_gravity[:, 2])[:, None]

  # Land position
  base_velocity = base_velocity_world_frame
  hip_velocity_body_frame = cross_quad(base_angular_velocity_body_frame,
                                       hip_positions_in_body_frame)
  hip_velocity = base_velocity[:, np.newaxis, :] + np.matmul(
      base_rot_mat, hip_velocity_body_frame.transpose(0, 2, 1)).transpose(0, 2, 1)


  land_position = hip_velocity * stance_duration[:, :, None] / 2
  land_position[..., 0] = np.clip(land_position[..., 0], -0.15, 0.15)
  land_position[..., 1] = np.clip(land_position[..., 1], -0.08, 0.08)
  land_position += hip_position
  land_position[..., 2] = (-base_height[:, None] + foot_landing_clearance)
  # -land_position[..., 0] * projected_gravity[:, 0, None]
  # -land_position[..., 1] * projected_gravity[:, 1, None]
  # ) / projected_gravity[:, 2, None]

  foot_position = _gen_swing_foot_trajectory(normalized_phase,
                                             phase_switch_foot_positions,
                                             mid_position, land_position)
  # print(f"Land position: {land_position}")
  # print(f"Foot position: {foot_position}")
  # ans = input("Any Key...")
  # if ans in ["Y", "y"]:
  #   import pdb
  #   pdb.set_trace()

  return foot_position


class RaibertSwingLegController:
  """Controls the swing leg position using Raibert's formula.
  For details, please refer to chapter 2 in "Legged robbots that balance" by
  Marc Raibert. The key idea is to stablize the swing foot's location based on
  the CoM moving speed.
  """
  def __init__(self,
               robot: Any,
               gait_generator: Any,
               desired_base_height: float = 0.26,
               foot_landing_clearance: float = -0.,
               foot_height: float = 0.15):
    self._robot = robot
    self._num_envs = self._robot.num_envs
    self._gait_generator = gait_generator
    self._last_leg_state = gait_generator.desired_contact_state
    self._foot_landing_clearance = foot_landing_clearance
    self._desired_base_height = desired_base_height
    self._foot_height = foot_height
    self._phase_switch_foot_positions = None
    self.reset()

  def reset(self) -> None:
    self._last_leg_state = np.copy(
        self._gait_generator.desired_contact_state)
    base_quat = self._robot.base_orientation_quat
    self._phase_switch_foot_positions = np.matmul(
    self._robot.base_rot_mat,
    np.transpose(self._robot.foot_positions_in_base_frame, (0, 2, 1))).transpose(0, 2, 1)


  def reset_idx(self, env_ids) -> None:
    self._last_leg_state[env_ids] = np.copy(
        self._gait_generator.desired_contact_state[env_ids])
    self._phase_switch_foot_positions[env_ids] = np.matmul(
        self._robot.base_rot_mat[env_ids],
        np.transpose(self._robot.foot_positions_in_base_frame[env_ids], (0, 2, 1))).transpose(0, 2, 1)

  def update(self) -> None:
    new_leg_state = np.copy(self._gait_generator.desired_contact_state)
    new_foot_positions = np.matmul(
        self._robot.base_rot_mat,
        np.transpose(self._robot.foot_positions_in_base_frame, (0, 2, 1))).transpose(0, 2, 1)
    self._phase_switch_foot_positions = np.where(
        np.tile((self._last_leg_state == new_leg_state)[:, :, None],
                   [1, 1, 3]), self._phase_switch_foot_positions,
        new_foot_positions)
    self._last_leg_state = new_leg_state

  @property
  def desired_foot_positions(self):
    """Computes desired foot positions in WORLD frame centered at robot base.

    Note: it returns an invalid position for stance legs.
    """
    return compute_desired_foot_positions(
        self._robot.base_rot_mat,
        self._robot.base_position[:, 2],
        self._robot.hip_positions_in_body_frame,
        self._robot.base_velocity_world_frame,
        self._robot.base_angular_velocity_body_frame,
        self._robot.projected_gravity,
        self._desired_base_height,
        self._foot_height,
        self._foot_landing_clearance,
        self._gait_generator.stance_duration,
        self._gait_generator.normalized_phase,
        self._phase_switch_foot_positions,
    )
