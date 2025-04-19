"""Computes leg states based on sinusoids and phase offsets."""
import copy
from typing import Any

from ml_collections import ConfigDict
import torch
import numpy as np

class PhaseGaitGenerator:
  """Computes desired gait based on leg phases."""
  def __init__(self, robot: Any, gait_config: ConfigDict):
    """Initializes the gait generator.
    Each gait is parameterized by 3 set of parameters:
      The _stepping frequency_: controls how fast the gait progresses.
      The _offset_: a 4-dim vector representing the offset from a standard
        gait cycle. In a standard gait cycle, each gait cycle starts in stance
        and ends in swing.
      The _swing ratio_: the percentage of air phase in each gait.
    """
    self._robot = robot
    self._num_envs = self._robot.num_envs
    self._config = ConfigDict()
    self._config.initial_offset = np.array(gait_config.initial_offset)  #convert to numpy array
    self._config.swing_ratio = np.array(gait_config.swing_ratio)   #convert to numpy array
    self._config.stepping_frequency = gait_config.stepping_frequency
    self.reset()

  def reset(self):
    self._current_phase = np.stack([self._config.initial_offset] *
                                      self._num_envs,
                                      axis=0)  #convert to numpy array
    self._stepping_frequency = np.ones(
        self._num_envs) * self._config.stepping_frequency  #convert to numpy array
    self._swing_cutoff = np.ones(
        (self._num_envs, 4)) * 2 * np.pi * (1 - self._config.swing_ratio) #convert to numpy array
    self._prev_frame_robot_time = self._robot.time_since_reset
    self._first_stance_seen = np.zeros((self._num_envs, 4),dtype=bool) #convert to numpy array

  def reset_idx(self, env_ids):
    self._current_phase[env_ids] = self._config.initial_offset
    self._stepping_frequency[env_ids] = self._config.stepping_frequency
    self._swing_cutoff[env_ids] = 2 * np.pi * (1 - self._config.swing_ratio)
    self._prev_frame_robot_time[env_ids] = self._robot.time_since_reset[
        env_ids]
    self._first_stance_seen[env_ids] = 0

  def update(self):
    current_robot_time = self._robot.time_since_reset
    delta_t = current_robot_time - self._prev_frame_robot_time
    self._prev_frame_robot_time = current_robot_time
    self._current_phase += 2 * np.pi * self._stepping_frequency[:,
                                                                   None] * delta_t[:,
                                                                                   None]
    

  @property
  def desired_contact_state(self):
    modulated_phase = np.remainder(self._current_phase + 2 * np.pi,
                                      2 * np.pi) #convert to numpy array
    raw_contact = np.where(modulated_phase > self._swing_cutoff, False,
                              True)  #convert to numpy array
    # print(f"Raw constact: {raw_contact}")
    self._first_stance_seen = np.logical_or(self._first_stance_seen,
                                               raw_contact) #convert to numpy array
    return np.where(self._first_stance_seen, raw_contact,
                       np.ones_like(raw_contact)) #convert to numpy array

  @property
  def desired_contact_state_se(self):
    """Also use odometry at the end of air phase."""
    modulated_phase = np.remainder(self._current_phase + 2 * np.pi,
                                      2 * np.pi) #convert to numpy array
    raw_contact = np.where(
        np.logical_and(modulated_phase > self._swing_cutoff,
                          modulated_phase < 2. * np.pi), False, True)  #convert to numpy array
    # print(f"Raw constact: {raw_contact}")
    self._first_stance_seen = np.logical_or(self._first_stance_seen,
                                               raw_contact) #convert to numpy array
    return np.where(self._first_stance_seen, raw_contact,
                       np.ones_like(raw_contact))  #convert to numpy array

  @property
  def normalized_phase(self):
    """Returns the leg's progress in the current state (swing or stance)."""
    modulated_phase = np.remainder(self._current_phase + 2 * np.pi,
                                      2 * np.pi)
    return np.where(modulated_phase < self._swing_cutoff,
                       modulated_phase / self._swing_cutoff,
                       (modulated_phase - self._swing_cutoff) /
                       (2 * np.pi - self._swing_cutoff))

  @property
  def stance_duration(self):
    return (self._swing_cutoff) / (2 * np.pi *
                                   self._stepping_frequency[:, None])

  @property
  def true_phase(self):
    return self._current_phase[:, 0] - self._config.initial_offset[0]

  @property
  def cycle_progress(self):
    true_phase = np.remainder(self.true_phase + 2 * np.pi, 2 * np.pi)
    return true_phase / (2 * np.pi)

  @property
  def stepping_frequency(self):
    return self._stepping_frequency

  @stepping_frequency.setter
  def stepping_frequency(self, new_frequency: np.ndarray):
    self._stepping_frequency = new_frequency    #might cause some issue
