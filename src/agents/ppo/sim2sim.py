# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os,sys
sys.path.append('./')
import math
import numpy as np
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from ml_collections import ConfigDict
# from src.controllers.sim2sim import adaptive_controller_numpy, AdaptiveController
from src.controllers.adaptive_controller import AdaptiveController
from src.robots.sim2sim.go1 import Go1
from src.robots.sim2sim.motors import MotorControlMode
from src.controllers.sim2sim import phase_gait_generator_numpy, raibert_swing_leg_controller_numpy, qp_torque_optimizer_numpy
import torch
import pinocchio as pin

class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
  cos_yaw = np.cos(robot_yaw)
  sin_yaw = np.sin(robot_yaw)
  world_frame_vec = gravity_frame_vec.copy()
  world_frame_vec[:, 0] = (cos_yaw * gravity_frame_vec[:, 0] -
                           sin_yaw * gravity_frame_vec[:, 1])
  world_frame_vec[:, 1] = (sin_yaw * gravity_frame_vec[:, 0] +
                           cos_yaw * gravity_frame_vec[:, 1])
  return world_frame_vec

def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
  cos_yaw = np.cos(robot_yaw)
  sin_yaw = np.sin(robot_yaw)
  gravity_frame_vec = world_frame_vec.copy()
  gravity_frame_vec[:, 0] = (cos_yaw * world_frame_vec[:, 0] +
                             sin_yaw * world_frame_vec[:, 1])
  gravity_frame_vec[:, 1] = (sin_yaw * world_frame_vec[:, 0] -
                             cos_yaw * world_frame_vec[:, 1])
  return gravity_frame_vec

class Sim2sim:
    def __init__(self,cfg):
        self.cfg = cfg.cfg
        self.sim_config = cfg.sim_config
        self._obs_buf = None
        self.num_envs = 1
        self._desired_landing_position = np.zeros((self.num_envs, 3))

        self._init_positions = np.zeros((self.num_envs, 3))
        self._init_positions[0,2] = 0.268
        self._robot = Go1(num_envs=self.num_envs,
                          init_positions=self._init_positions,
                          sim_config=cfg.sim_config,
                          motor_control_mode=MotorControlMode.HYBRID,
                          motor_torque_delay_steps=self.cfg.get('motor_torque_delay_steps', 0)) 
        
        self._robot.set_foot_frictions(self.cfg.get('foot_friction', np.array([5., 5., 5., 5.])))
        self._gait_generator = phase_gait_generator_numpy.PhaseGaitGenerator(self._robot, cfg.gait)
        self._swing_leg_controller = raibert_swing_leg_controller_numpy.RaibertSwingLegController(
                self._robot, self._gait_generator, foot_height=self.cfg.swing_foot_height, 
                foot_landing_clearance=self.cfg.swing_foot_landing_clearance)
        
        # self._torque_optimizer = qp_torque_optimizer_numpy.QPTorqueOptimizer(
        #     self._robot,
        #     base_position_kp=self.cfg.get('base_position_kp', np.array([0., 0., 50.])),
        #     base_position_kd=self.cfg.get('base_position_kd', np.array([10., 10., 10.])),
        #     base_orientation_kp=self.cfg.get('base_orientation_kp', np.array([50., 50., 0.])),
        #     base_orientation_kd=self.cfg.get('base_orientation_kd', np.array([10., 10., 10.])),
        #     weight_ddq=self.cfg.get('qp_weight_ddq', np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
        #     foot_friction_coef=self.cfg.get('qp_foot_friction_coef', 0.7),
        #     clip_grf=self.cfg.get('clip_grf_in_sim'),
        #     body_inertia=self.cfg.get('qp_body_inertia', np.array([0.14, 0.35, 0.35]) * 0.5),
        #     use_full_qp=self.cfg.get('use_full_qp', False))
        
        self._adaptive_ctrl = AdaptiveController(
            robot            = self._robot,
            kp               = self.cfg.get('adaptive_kp',   np.ones(12)*50.0),
            kd               = self.cfg.get('adaptive_kd',   np.ones(12)*1.0),
            gamma            = self.cfg.get('adaptive_gamma',np.ones( 1)*10.0),  # e.g. single scalar param
            lambda_param     = self.cfg.get('adaptive_lambda',np.ones( 1)*5.0),
            num_robot_params = 1,                                  # how many θ’s you adapt
            dt               = self._robot.control_timestep)
        
        self._init_yaw = np.zeros((self.num_envs,))
        self._steps_count = np.zeros((self.num_envs,))
        self._construct_observation_and_action_space()
        self._cycle_count = np.zeros((self.num_envs,))
        self._jumping_distance = np.zeros((self.num_envs, 2))
        self._desired_landing_position[0,2] = 0.268
        self._jumping_distance[0,0] = 0.5
        self._jumping_distance[0,1] = 0.0
        self._resample_command(np.arange(self.num_envs))
    ## TODO: set the action space in cfg file
    def _construct_observation_and_action_space(self):
        robot_lb = np.array([0., -3.14, -3.14, -4., -4., -10., -3.14, -3.14, -3.14] +
                            [-0.5, -0.5, -0.4] * 4)
        robot_ub = np.array([0.6, 3.14, 3.14, 4., 4., 10., 3.14, 3.14, 3.14] +
                            [0.5, 0.5, 0.] * 4)
        task_lb = np.array([-2., -2., -1., -1., -1.])
        task_ub = np.array([2., 2., 1., 1., 1.])
        self._observation_lb = np.concatenate((task_lb, robot_lb))
        self._observation_ub = np.concatenate((task_ub, robot_ub))
        self._actions_lb = np.array(self.cfg.action_lb)
        self._actions_ub = np.array(self.cfg.action_ub)

    def _normalize_obs(self,obs):
        min_ = self._observation_lb
        max_ = self._observation_ub
        obs = 2 * (obs - min_) / (max_ - min_) - 1
        return obs
    
    def _denormalize_action(self,action):
        min_ = self._actions_lb
        max_ = self._actions_ub
        action = (action + 1) / 2 * (max_ - min_) + min_
        return action
    
    def _resample_command(self,env_ids):
        
        self._jumping_distance[env_ids,0] = 0.3
        self._jumping_distance[env_ids,1] = 0.0
        self._desired_landing_position[env_ids, :2] = self._robot.base_position[
        env_ids, :2] + gravity_frame_to_world_frame(
            self._robot.base_orientation_rpy[env_ids, 2],
            self._jumping_distance[env_ids])
        self._desired_landing_position[env_ids, 2] = 0.268

    def _split_action(self,action):
        gait_action = None
        if self.cfg.get('include_gait_action', False):
            gait_action = action[:, :1]
            action = action[:, 1:]

        foot_action = None
        if self.cfg.get('include_foot_action', False):
            if self.cfg.get('mirror_foot_action', False):
                foot_action = action[:, -6:].reshape(
                    (-1, 2, 3))  #+ self._robot.hip_offset
                foot_action = np.stack([
                    foot_action[:, 0],
                    foot_action[:, 0],
                    foot_action[:, 1],
                    foot_action[:, 1],
        ],
                                  axis=1)
                action = action[:, :-6]
            else:
                foot_action = action[:, -12:].reshape(
                    (-1, 4, 3))  #+ self._robot.hip_offset
                action = action[:, :-12]

        com_action = action
        return gait_action, com_action, foot_action
    
    def reset(self):
        
        return self._normalize_obs(self.reset_idx(np.arange(self.num_envs)))
    
    def reset_idx(self,env_ids):
        self._obs_buf = self.get_obs()
        self._steps_count[env_ids] = 0
        self._cycle_count[env_ids] = 0
        self._init_yaw[env_ids] = self._robot.base_orientation_rpy[env_ids, 2]
        
        self._robot.reset_idx(env_ids)
        self._swing_leg_controller.reset_idx(env_ids)
        self._gait_generator.reset_idx(env_ids)
        self._resample_command(env_ids)
        
        return self._obs_buf
    
    def wrapToPi(self, x):
        return np.remainder(x + np.pi, 2 * np.pi) - np.pi

    def step(self, action):
        action = self._denormalize_action(action)
        action = np.clip(action, self._actions_lb, self._actions_ub)
        gait_action, com_action, foot_action = self._split_action(action)
        zero = np.zeros((self.num_envs,))

        n_substeps = max(int(self.cfg.env_dt / self._robot.control_timestep), 1)
        for _ in range(n_substeps):
            self._gait_generator.update()
            self._swing_leg_controller.update()
            if gait_action is not None:
                self._gait_generator.stepping_frequency = gait_action[:, 0]

            base_z       = self._robot.base_position[:, 2]
            base_vz      = self._robot.base_velocity_body_frame[:, 2]
            pitch        = self._robot.base_orientation_rpy[:, 1]
            yaw          = self._robot.base_orientation_rpy[:, 2]
            omega_y      = self._robot.base_angular_velocity_body_frame[:, 1]

            z_err       = com_action[:, 0] - base_z
            zdot_err    = com_action[:, 2] - base_vz
            vert_acc    = self.cfg.base_kp_z * z_err + self.cfg.base_kd_z * zdot_err

            pitch_err   = com_action[:, 5] - pitch
            pdot_err    = com_action[:, 6] - omega_y
            pitch_acc   = self.cfg.base_kp_pitch * pitch_err + self.cfg.base_kd_pitch * pdot_err

            desired_yaw      = yaw.copy()
            desired_yaw_rate = zero.copy()
            if self.cfg.get("use_yaw_feedback", False):
                raw        = com_action[:, 7] - yaw
                yaw_err    = (raw + np.pi) % (2 * np.pi) - np.pi
                desired_yaw      = yaw + yaw_err
                desired_yaw_rate = yaw_err

            desired_foot_positions = self._swing_leg_controller.desired_foot_positions.copy()
            if foot_action is not None:
                cosy = np.cos(yaw)[:, None]
                siny = np.sin(yaw)[:, None]
                fw   = foot_action.copy()
                fw[:, :, 0] =  cosy * foot_action[:, :, 0] - siny * foot_action[:, :, 1]
                fw[:, :, 1] =  siny * foot_action[:, :, 0] + cosy * foot_action[:, :, 1]
                desired_foot_positions += fw

            q_joints_np = self._robot.get_motor_angles_from_foot_positions(
                desired_foot_positions)
            q_joints    = torch.from_numpy(q_joints_np).float()
            qd_joints   = torch.zeros_like(q_joints)

            B = self.num_envs
            q_ref   = torch.zeros(B, 18)
            qd_ref  = torch.zeros(B, 18)
            qdd_ref = torch.zeros(B, 18)

            q_ref[:, 2]   = torch.from_numpy(com_action[:, 0])
            qd_ref[:, 2]  = torch.from_numpy(com_action[:, 2])
            qdd_ref[:, 2] = torch.from_numpy(vert_acc)

            q_ref[:, 4]   = torch.from_numpy(com_action[:, 5])
            qd_ref[:, 4]  = torch.from_numpy(com_action[:, 6])
            qdd_ref[:, 4] = torch.from_numpy(pitch_acc)

            q_ref[:, 5]   = torch.from_numpy(desired_yaw)
            qd_ref[:, 5]  = torch.from_numpy(desired_yaw_rate)

            q_ref[:, 6:]  = q_joints
            qd_ref[:, 6:] = qd_joints
            # leave qdd_ref[:,6:] = 0

            motor_torques = self._adaptive_ctrl.get_action(q_ref, qd_ref, qdd_ref)

            self._robot.step(motor_torques)

            self._obs_buf = self.get_obs()
            new_cycles  = np.floor(self._gait_generator.true_phase / (2 * np.pi)).astype(int)
            done_envs   = np.where(new_cycles > self._cycle_count)[0]
            self._resample_command(done_envs)
            self._cycle_count = new_cycles

        return self._normalize_obs(self._obs_buf)
        
    def get_obs(self):
        distance_to_goal = self._desired_landing_position - self._robot.base_position_world

        distance_to_goal_local = world_frame_to_gravity_frame(
            self._robot.base_orientation_rpy[:, 2], distance_to_goal)
        phase_obs = np.stack((
            np.cos(self._gait_generator.true_phase),
            np.sin(self._gait_generator.true_phase),
        ),
                                axis=1)

        robot_obs = np.concatenate(
            (
                self._robot.base_position[:, 2:],  # Base height
                self._robot.base_orientation_rpy[:, 0:1] * 0,  # Base roll
                self._robot.base_orientation_rpy[:, 1:2],  # Base Pitch
                self._robot.base_velocity_body_frame[:, 0:1],
                self._robot.base_velocity_body_frame[:, 1:2] * 0,
                self._robot.base_velocity_body_frame[:, 2:3],  # Base velocity (z)
                self._robot.base_angular_velocity_body_frame[:, 0:1] * 0,
                self._robot.base_angular_velocity_body_frame[:, 1:2],
                self._robot.base_angular_velocity_body_frame[:,
                                                            2:3],  # Base yaw rate
                # self._robot.motor_positions,
                # self._robot.motor_velocities,
                self._robot.foot_positions_in_base_frame.reshape(
                    (self.num_envs, 12)),
            ),
            axis=1)
        obs = np.concatenate((distance_to_goal_local, phase_obs, robot_obs),
                                axis=1)
        if self.cfg.get("observation_noise",
                            None) is not None:
            obs += np.random.randn_like(obs) * self.cfg.observation_noise
        # print("foot_pos",np.round(self._robot.foot_positions_in_base_frame,2))
        return obs

    def run_mujoco(self,policy):
        """
        Run the Mujoco simulation using the provided policy and configuration.

        Args:
            policy: The policy used for controlling the simulation.
            cfg: The configuration object containing simulation settings.

        Returns:
            None
        """
        step_count = 0
       
        state = self.reset()
        
        # exit()
        for _ in tqdm(range(int(self.sim_config.sim_duration / self.sim_config.dt)), desc="Simulating..."):
            step_count += 1

            action = policy(torch.tensor(state.astype(np.float32))).detach().numpy()
            
            # exit()
            # import ipdb; ipdb.set_trace()
            
            state = self.step(action)
            # print("state")
            # print(np.round(state,2))
            # exit()    

        self._robot.viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg():

        sim_config=ConfigDict() 
        sim_config.sim_duration = 60.0
        sim_config.dt = 0.002  
        sim_config.decimation = 10
        sim_config.render = True
        sim_config.action_repeat = 1
        gait = ConfigDict()
        gait.stepping_frequency = 1
        gait.initial_offset = np.array([0., 0., 0., 0.]) * (2 * np.pi)
        gait.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5])
        cfg = ConfigDict()
        cfg.goal_lb = np.array([0.3, 0.])
        cfg.goal_ub = np.array([1., 0.])

        cfg.include_gait_action = True
        cfg.include_foot_action = True
        cfg.mirror_foot_action = True

        cfg.action_lb = np.array([0.5, -0.001, -3, -0.001, -3., -0.001, -0.001, -2.5, -0.001] +
                                [-0.1, -0.0001, 0.] * 2)
        cfg.action_ub = np.array([3.999, 0.001, 3, 0.001, 3., 0.001, 0.001, 2.5, 0.001] +
                               [0.1, 0.001, 0.2] * 2)
        
        cfg.episode_length_s = 20.
        cfg.max_jumps = 10.
        cfg.env_dt = 0.01
        cfg.motor_strength_ratios = 1.
        cfg.motor_torque_delay_steps = 5
        cfg.use_yaw_feedback = False
        cfg.foot_friction = [1., 1., 1., 1.]  #0.7
        cfg.base_position_kp = np.array([0., 0., 0.])
        cfg.base_position_kd = np.array([10., 10., 10.])
        cfg.base_kp_z = cfg.base_position_kp[2]
        cfg.base_kd_z = cfg.base_position_kd[2]
        cfg.base_orientation_kp = np.array([50., 0., 0.])
        cfg.base_orientation_kd = np.array([10., 10., 10.])
        cfg.base_kp_pitch = cfg.base_orientation_kp[1]
        cfg.base_kd_pitch = cfg.base_orientation_kd[1]
        cfg.qp_foot_friction_coef = 0.6
        cfg.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
        cfg.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5
        cfg.use_full_qp = False
        cfg.clip_grf_in_sim = True
        cfg.swing_foot_height = 0.
        cfg.swing_foot_landing_clearance = 0.
        cfg.terminate_on_body_contact = True
        cfg.terminate_on_limb_contact = False
        cfg.terminate_on_height = 0.15
        cfg.use_penetrating_contact = False


    policy = torch.jit.load(args.load_model)
    sim2sim = Sim2sim(Sim2simCfg())
    sim2sim.run_mujoco(policy)