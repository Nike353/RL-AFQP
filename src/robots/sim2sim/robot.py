"""General class for (vectorized) robots."""
import os
import sys
from typing import Any, List


import ml_collections
import numpy as np
import torch
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
from src.utilities.rotation_utils import  get_euler_xyz_from_quaternion_np, quat_to_rot_mat_np

def angle_normalize(x):
    return np.remainder(x + np.pi, 2 * np.pi) - np.pi


class Robot:
  """General class for simulated quadrupedal robot."""
  def __init__(
      self,
      init_positions: np.ndarray,
      xml_path: str,
      sim_config: ml_collections.ConfigDict,
      motors: Any,
      feet_names: List[str],
      calf_names: List[str],
      thigh_names: List[str],
  ):
    """Initializes the robot class."""
    
    self._sim_config = sim_config
    self._motors = motors
    self._feet_names = feet_names
    self._calf_names = calf_names
    self._thigh_names = thigh_names

    self._base_init_state = self._compute_base_init_state(init_positions)
    self._init_motor_angles = self._motors.init_positions

    self._load_xml(xml_path)
    self.num_envs = 1
    self._init_buffers()
    
    self._time_since_reset = np.zeros(1)
    # self.reset()
    self._post_physics_step()
    
    # self.reset()

  def _compute_base_init_state(self, init_positions: np.ndarray):
    """Computes desired init state for CoM (position and velocity)."""
    num_envs = init_positions.shape[0]
    init_state_list = [0., 0., 0.] + [1., 0., 0., 0.] + [0., 0., 0.] + [0., 0., 0.]   #compos quat comvel angvel
    # init_state_list = [0., 0., 0.] + [0., 0., 0.7071, 0.7071] + [0., 0., 0.
    #                                                      ] + [0., 0., 0.]
    # init_state_list = [0., 0., 0.] + [ 0.0499792, 0, 0, 0.9987503
    #                                       ] + [0., 0., 0.] + [0., 0., 0.]
    init_states = np.stack([init_state_list] * num_envs, axis=0)
    init_states[:, :3] = init_positions
    return init_states

  def _load_xml(self, xml_path):
    self.model = mujoco.MjModel.from_xml_path(xml_path)
    self.model.opt.timestep = self._sim_config.dt
    self.data = mujoco.MjData(self.model)
    # mujoco.mj_step(self.model, self.data)
    # print("self.data.xpos")
    # print(self.data.xpos)
    # print("self.data.qpos")
    # print(self.data.qpos)
    
    
    # self.viewer = 
    self.viewer = mujoco.viewer.launch_passive(self.model,self.data,
                                               show_left_ui=False,
                                               show_right_ui=False,
                                               key_callback=self.viewer_key_callback)
    
     # visual markers
    self.vis_markers = []
    # manipulatable camera
    self.free_camera = mujoco.MjvCamera()
    # default_qpos = np.array([0.0,0.0,0.27,1.0,0.0,0.0,0.0,0.0,0.9,-1.8,0.0,0.9,-1.8,0.0,0.9,-1.8,0.0,0.9,-1.8])
    # self.data.qpos = default_qpos.copy()
    tree = ET.parse(xml_path)
    self.viewer_paused = True
    root = tree.getroot()
    kf_element = root.find(".//key[@name='home']")
    if kf_element is not None and 'qpos' in kf_element.attrib:
        key_frame_str = kf_element.attrib['qpos']
        key_frame_qpos = np.fromstring(key_frame_str, sep=' ')
        
        if key_frame_qpos.size != self.model.nq:
            raise ValueError(
                f"The extracted qpos has {key_frame_qpos.size} values, but expected {self.model.nq}."
            )
        
        self.data.qpos[:] = key_frame_qpos
        mujoco.mj_forward(self.model, self.data)
    else:
        self.data.qpos[:] = [0,0,0.27,1,0,0,0,0,0.9,-1.8,0,0.9,-1.8,0,0.9,-1.8,0,0.9,-1.8]
        mujoco.mj_forward(self.model, self.data)
        print("No keyframe element with attribute 'qpos' found; using default qpos values.")
    self.viewer.sync()
    print("hi")
    self._num_dof = self.model.nu
    self._num_bodies = self.model.nbody
    self._body_indices = []
    self._body_names = []
    self._feet_indices = []
    self._calf_indices = []
    self._thigh_indices = []

    for i in range(self.model.nbody):
      body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
      print(body_name)
      if body_name in self._feet_names:
        self._feet_indices.append(i)
      elif body_name in self._calf_names:
        self._calf_indices.append(i)
      elif body_name in self._thigh_names:
        self._thigh_indices.append(i)
      else:
        self._body_indices.append(i)
        self._body_names.append(body_name)
    
  ###viewer related functions  
  def viewer_key_callback(self, keycode):
    if chr(keycode) == ' ':
            # print('space')
            self.viewer_paused = not self.viewer_paused
    elif chr(keycode) == 'E':
            self.viewer.opt.frame = not self.viewer.opt.frame
  def _init_buffers(self):
    # get gym GPU state tensors
    

    # Robot state buffers
    self._root_states = np.zeros((1, 13))
    self._root_states[:, :3] = self.data.qpos[:3]
    self._root_states[:, 3:7] = self.data.qpos[3:7]
    self._root_states[:, 7:10] = self.data.qvel[:3]
    self._root_states[:, 10:13] = self.data.qvel[3:6]

    self._motor_positions = self.data.qpos[7:].reshape(1,self.num_dof)
    self._motor_velocities = self.data.qvel[6:].reshape(1,self.num_dof)
    self._base_quat = self._root_states[:, 3:7]
    self._base_rot_mat = np.array(quat_to_rot_mat_np(self._base_quat))
    self._base_rot_mat_t = np.transpose(self._base_rot_mat, (0,1, 2))

    self._base_lin_vel_world = self._root_states[:, 7:10]
    self._base_ang_vel_world = self._root_states[:, 10:13]
    self._gravity_vec = np.array([[0., 0., 1.]])  # Shape (1, 3)

    self._projected_gravity = np.matmul(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]
    

    ''' TODO: have to attach a sensor to get the pos and vel of the feet'''
    # self._foot_velocities = self._rigid_body_state.view(
    #     1, self._num_bodies, 13)[:, self._feet_indices, 7:10]
    # self._foot_positions = self._rigid_body_state.view(1,
    #                                                    self._num_bodies,
    #                                                    13)[:,
    #                                                        self._feet_indices,
    #                                                        0:3]
    # Other useful buffers
    self._foot_positions = self.data.xpos[self._feet_indices].reshape(1, 4, 3)
    self._foot_velocities = self.data.cvel[self._feet_indices].reshape(1, 4, 6)[:,:,3:]
    self._torques = np.zeros((1,
                                self._num_dof)
                                )
    self._dof_state = np.zeros((1, self._num_dof, 2))
    self._get_jacobian()
    # print(self.data.xpos)
    # print(self.data.qpos)
    # exit()
  def _get_jacobian(self):
    self._jacobian = np.zeros((self.num_envs, self._num_bodies, 6, self.model.nv))
    for i in range(self.model.nbody):
      Jp = np.zeros((3, self.model.nv))
      Jr = np.zeros((3, self.model.nv))
      mujoco.mj_jac(self.model, self.data, Jp, Jr, self.data.xpos[i],i)
      self._jacobian[:, i, :3, :] = Jp
      self._jacobian[:, i, 3:, :] = Jr
    
  def reset(self):
    self.reset_idx(np.arange(1))

  def reset_idx(self, env_ids):
    if len(env_ids) == 0:
      return
    self._time_since_reset[env_ids] = 0
    # Reset root states:
    self._root_states[env_ids] = self._base_init_state[env_ids]

    self.data.qpos[:7] = self._root_states[env_ids][:,:7]
    self.data.qvel[:6] = self._root_states[env_ids][:,7:]

    # Reset dofs
    self._motor_positions[env_ids] = self._init_motor_angles
    self._motor_velocities[env_ids] = 0.

    self.data.qpos[7:] = self._motor_positions[env_ids]
    self.data.qvel[6:] = self._motor_velocities[env_ids]

    
    mujoco.mj_forward(self.model,self.data)   
   
    # print(self.data.cvel)
    self._post_physics_step()
    
  
     

  def step(self, action):
    if not self.viewer_paused:
      for _ in range(self._sim_config.action_repeat):
        self._torques, _ = self.motor_group.convert_to_torque(
            action, self._motor_positions, self._motor_velocities)
        self.data.ctrl[:] = self._torques
        mujoco.mj_step(self.model, self.data)
        self._time_since_reset += self._sim_config.dt
      # if self._sim_config.render:
        # self.viewer.render()
        # mujoco.viewer.sync()
      self.viewer.sync()
      self._root_states[:, :3] = self.data.qpos[:3]
      self._root_states[:, 3:7] = self.data.qpos[3:7]
      self._root_states[:, 7:10] = self.data.qvel[:3]
      self._root_states[:, 10:13] = self.data.qvel[3:6]
      self._post_physics_step()
      self.viewer.cam.lookat[0] = self.data.qpos[0]
      self.viewer.cam.lookat[1] = self.data.qpos[1]
      self.viewer.cam.lookat[2] = 0.5
      self.viewer.cam.azimuth = 90

  def _post_physics_step(self):
    
    self._base_quat[:] = self._root_states[:, 3:7]
    self._base_rot_mat = quat_to_rot_mat_np(self._base_quat)

    self._base_rot_mat_t = np.transpose(self._base_rot_mat, (0,1, 2))
    self._base_lin_vel_world = self._root_states[:, 7:10]
    self._base_ang_vel_world = self._root_states[:, 10:13]
    self._projected_gravity[:] = np.matmul(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]
    
    ''' TODO: have to attach a sensor to get the pos and vel of the feet'''
    # self._foot_velocities = self._rigid_body_state.view(
    #     1, self._num_bodies, 13)[:, self._feet_indices, 7:10]
    # self._foot_positions = self._rigid_body_state.view(1,
    #                                                    self._num_bodies,
    #                                                    13)[:,
    #                                                        self._feet_indices,
    #   
   
    self._foot_positions = self.data.xpos[self._feet_indices].reshape(1, 4, 3)
    self._foot_velocities = self.data.cvel[self._feet_indices].reshape(1, 4, 6)[:,:,3:]
    # print("self._foot_positions")
    # print(self._foot_positions)
    # print("self._foot_velocities")
    # print(self._foot_velocities)
    # exit()
    
    # exit()

  def get_motor_angles_from_foot_positions(self, foot_local_positions):
    raise NotImplementedError()

  def update_init_positions(self, env_ids, init_positions):
    self._base_init_state[env_ids] = self._compute_base_init_state(
        init_positions)

  def set_foot_frictions(self,friction_coefs):
    foot_geoms = ["FR", "FL", "RR", "RL"]
    for i, geom in enumerate(foot_geoms):
      geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
      self.model.geom_friction[geom_id][0] = friction_coefs[i]
    
  @property
  def base_position(self):
    base_position = self._root_states[:, :3].copy()
    return base_position

  @property
  def base_position_world(self):
    return self._root_states[:, :3]

  @property
  def base_orientation_rpy(self):
    return angle_normalize(
        np.array(get_euler_xyz_from_quaternion_np(self._root_states[:, 3:7])))

  @property
  def base_orientation_quat(self):
    return self._root_states[:, 3:7]

  @property
  def projected_gravity(self):
    return self._projected_gravity

  @property
  def base_rot_mat(self):
    return self._base_rot_mat

  @property
  def base_rot_mat_t(self):
    return self._base_rot_mat_t

  @property
  def base_velocity_world_frame(self):
    return self._base_lin_vel_world

  @property
  def base_velocity_body_frame(self):
    return np.matmul(self._base_rot_mat_t, self._root_states[:, 7:10,
                                                             None])[:, :, 0]

  @property
  def base_angular_velocity_world_frame(self):
    return self._base_ang_vel_world

  @property
  def base_angular_velocity_body_frame(self):
    return np.matmul(self._base_rot_mat_t, self._root_states[:, 10:13,
                                                             None])[:, :, 0]

  @property
  def motor_positions(self):
    return self._motor_positions.copy()

  @property
  def motor_velocities(self):
    return self._motor_velocities.copy()

  @property
  def motor_torques(self):
    return self._torques.copy()

  ''' TODO: have to attach a sensor to get the pos and vel of the feet'''
  @property
  def foot_positions_in_base_frame(self):
    foot_positions_world_frame = self._foot_positions
    base_position_world_frame = self._root_states[:, :3]
    # num_env x 4 x 3
    foot_position = (foot_positions_world_frame -
                     base_position_world_frame[:, None, :])
    # print("foot_position_world_frame")
    # print(foot_positions_world_frame)
    # print("base_position_world_frame")
    # print(base_position_world_frame)
    # print("foot_position")
    # print(foot_position)
    # print("self._base_rot_mat_t")
    # print(self._base_rot_mat_t)
    # exit()
    return np.matmul(self._base_rot_mat_t, foot_position.transpose(0, 2, 1)).transpose(0, 2, 1)

  @property
  def motor_group(self):
    return self._motors

  @property
  def num_dof(self):
    return self._num_dof

  @property
  def control_timestep(self):
    return self._sim_config.dt * self._sim_config.action_repeat

  @property
  def time_since_reset(self):
    return self._time_since_reset.copy()
  
  @property
  def all_foot_jacobian(self):
    rot_mat_t = self.base_rot_mat_t  # Shape: (num_envs, 3, 3)
    jacobian = np.zeros((self.num_envs, 12, 12))
    jacobian[:, :3, :3] = np.matmul(rot_mat_t, self._jacobian[:, self._feet_indices[0], :3, 6:9])
    jacobian[:, 3:6, 3:6] = np.matmul(rot_mat_t, self._jacobian[:, self._feet_indices[1], :3, 9:12])
    jacobian[:, 6:9, 6:9] = np.matmul(rot_mat_t, self._jacobian[:, self._feet_indices[2], :3, 12:15])
    jacobian[:, 9:12, 9:12] = np.matmul(rot_mat_t, self._jacobian[:, self._feet_indices[3], :3, 15:18])
    
    return jacobian

  
 

  @property
  def foot_positions_in_world_frame(self):
    return self._foot_positions.copy()

  @property
  def foot_height(self):
    return self._foot_positions[:, :, 2]

  # @property
  # def foot_velocities_in_base_frame(self):
  #   foot_vels = np.squeeze(np.matmul(self.all_foot_jacobian, self.motor_velocities[:, :, None]))

  #   return foot_vels.reshape((1, 4, 3))

  @property
  def foot_velocities_in_world_frame(self):
    return self._foot_velocities