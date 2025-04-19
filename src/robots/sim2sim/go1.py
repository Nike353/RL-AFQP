"""Vectorized Go1 robot in Isaac Gym."""
from typing import Any, Sequence, Union

import ml_collections
import torch
import numpy as np

from src.robots.sim2sim.robot import Robot
from src.robots.sim2sim.motors import MotorControlMode, MotorGroup, MotorModel

_ARRAY = Sequence[float]


def motor_angles_from_foot_positions(foot_local_positions,
                                     hip_offset,
                                     device: str = "cuda"):
  foot_positions_in_hip_frame = foot_local_positions - hip_offset
  l_up = 0.213
  l_low = 0.233
  l_hip = 0.08 * np.array([-1, 1, -1, 1])

  x = foot_positions_in_hip_frame[:, :, 0]
  y = foot_positions_in_hip_frame[:, :, 1]
  z = foot_positions_in_hip_frame[:, :, 2]
  theta_knee = -np.arccos(
      np.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                 (2 * l_low * l_up), -1, 1))
  l = np.sqrt(
      np.clip(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee),
                 1e-7, 1))
  theta_hip = np.arcsin(np.clip(-x / l, -1, 1)) - theta_knee / 2
  c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
  s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)

  # thetas: num_envs x 4
  joint_angles = np.stack([theta_ab, theta_hip, theta_knee], axis=2)
  return joint_angles.reshape((-1, 12))


class Go1(Robot):
  """Go1 robot in simulation."""

  def __init__(
      self,
      sim_config: ml_collections.ConfigDict(),
      num_envs: int,
      init_positions: _ARRAY,
      motor_control_mode: MotorControlMode,
      motor_torque_delay_steps: int = 0,
  ):

    motors = MotorGroup(
                        num_envs=num_envs,
                        motors=(
                            
                            
                            MotorModel(
                                name="FR_hip_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.0,
                                min_position=-0.802851455917,
                                max_position=0.802851455917,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="FR_thigh_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.9,
                                min_position=-1.0471975512,
                                max_position=4.18879020479,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="FR_calf_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=-1.8,
                                min_position=-2.6965336943,
                                max_position=-0.916297857297,
                                min_velocity=-20,
                                max_velocity=20,
                                min_torque=-35.55,
                                max_torque=35.55,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="FL_hip_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.0,
                                min_position=-0.802851455917,
                                max_position=0.802851455917,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="FL_thigh_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.9,
                                min_position=-1.0471975512,
                                max_position=4.18879020479,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="FL_calf_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=-1.8,
                                min_position=-1.0471975512,
                                max_position=4.18879020479,
                                min_velocity=-20,
                                max_velocity=20,
                                min_torque=-35.55,
                                max_torque=35.55,
                                kp=100,
                                kd=1,
                            ),
                            
                            MotorModel(
                                name="RR_hip_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.0,
                                min_position=-0.802851455917,
                                max_position=0.802851455917,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="RR_thigh_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.9,
                                min_position=-1.0471975512,
                                max_position=4.18879020479,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="RR_calf_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=-1.8,
                                min_position=-2.6965336943,
                                max_position=-0.916297857297,
                                min_velocity=-20,
                                max_velocity=20,
                                min_torque=-35.55,
                                max_torque=35.55,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="RL_hip_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.0,
                                min_position=-0.802851455917,
                                max_position=0.802851455917,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="RL_thigh_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=0.9,
                                min_position=-1.0471975512,
                                max_position=4.18879020479,
                                min_velocity=-30,
                                max_velocity=30,
                                min_torque=-23.7,
                                max_torque=23.7,
                                kp=100,
                                kd=1,
                            ),
                            MotorModel(
                                name="RL_calf_joint",
                                motor_control_mode=motor_control_mode,
                                init_position=-1.8,
                                min_position=-2.6965336943,
                                max_position=-0.916297857297,
                                min_velocity=-20,
                                max_velocity=20,
                                min_torque=-35.55,
                                max_torque=35.55,
                                kp=100,
                                kd=1,
                            ),
                        ),
                        torque_delay_steps=motor_torque_delay_steps)

    com_offset = -np.array([0.011611, 0.004437, 0.000108])
    self._hip_offset = np.array(
        [[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.], [-0.1881, -0.04675, 0.],
         [-0.1881, 0.04675, 0.]]) + com_offset

    delta_x, delta_y = 0.0, 0.0
    hip_position_single = np.array((
        (0.1835 + delta_x, -0.131 - delta_y, 0),
        (0.1835 + delta_x, 0.122 + delta_y, 0),
        (-0.1926 - delta_x, -0.131 - delta_y, 0),
        (-0.1926 - delta_x, 0.122 + delta_y, 0),
    ))
    self._hip_positions_in_body_frame = np.stack([hip_position_single] *
                                                    num_envs,
                                                    axis=0)

    super().__init__(
                     init_positions=init_positions,
                     xml_path="data/go1/xml/scene.xml",
                     sim_config=sim_config,
                     motors=motors,
                     feet_names=[
                         "FR_foot",
                         "FL_foot",
                         "RR_foot",
                         "RL_foot",
                     ],
                     calf_names=[
                         "FR_calf",
                         "FL_calf",
                         "RR_calf",
                         "RL_calf",
                     ],
                     thigh_names=[
                         "FR_thigh",
                         "FL_thigh",
                         "RR_thigh",
                         "RL_thigh",
                     ])

  @property
  def hip_positions_in_body_frame(self):
    return self._hip_positions_in_body_frame

  @property
  def hip_offset(self):
    """Position of hip offset in base frame, used for IK only."""
    return self._hip_offset

  def get_motor_angles_from_foot_positions(self, foot_local_positions):
    return motor_angles_from_foot_positions(foot_local_positions,
                                            self.hip_offset,
                                            )