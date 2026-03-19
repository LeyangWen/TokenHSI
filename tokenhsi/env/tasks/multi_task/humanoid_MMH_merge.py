# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import os
from enum import Enum
import numpy as np
import torch
import yaml
import trimesh
import pickle
import wandb

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *
from env.tasks.basic_interaction_skills.humanoid_carry import compute_back_ergo_reward, compute_box_ergo_reward, compute_elbow_ergo_reward
from env.tasks.basic_interaction_skills.humanoid_carry import compute_handheld_reward, compute_walk_reward, compute_carry_reward, compute_putdown_reward
from env.tasks.basic_interaction_skills.humanoid_carry import compute_handheld_timber_reward, compute_handheld_bag_reward, compute_handheld_handle_reward
from env.tasks.basic_interaction_skills.humanoid_carry import compute_humanoid_reset, build_amp_observations, compute_location_observations
from utils import torch_utils

class HumanoidMMHMerge(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    class TaskUID(Enum):
        MMH_box = 0
        MMH_handle = 1
        MMH_timber = 2
        MMH_bag = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._multiple_task_names = ["MMH_box", "MMH_handle", "MMH_timber", "MMH_bag"]
        self._num_tasks = len(self._multiple_task_names)

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._only_vel_reward = cfg["env"]["onlyVelReward"]
        self._only_height_handheld_reward = cfg["env"]["onlyHeightHandHeldReward"]
        self._enable_task_mask_obs = cfg["env"].get("enableTaskMaskObs", True)
        self._enable_task_specific_disc = cfg["env"].get("enableTaskSpecificDisc", True)

        self._box_vel_penalty = cfg["env"]["box_vel_penalty"]
        self._box_vel_pen_coeff = cfg["env"]["box_vel_pen_coeff"]
        self._box_vel_pen_thre = cfg["env"]["box_vel_pen_threshold"]

        self._mode = cfg["env"]["mode"]
        assert self._mode in ["train", "test"]

        self._is_eval = cfg["args"].eval
        self._is_test = cfg["args"].test
        self.constructionExp = cfg["env"]["eval"].get("constructionExperiment", False)
        self._box_density_value = cfg["env"]["eval"].get("density", False)
        print(f"[Info]: Value or False: _box_density_value = {self._box_density_value}")

        self._ergo_coeff = cfg["env"].get("ergoCoeff", False)
        self._ergo_sub_weight = cfg["env"].get("ergoSubWeight", False)
        self._verbose = False

        if cfg["args"].eval:
            self._mode = "test"

        # configs for physical box actor (shared by all MMH tasks in this merged env)
        box_cfg = cfg["env"]["box"]
        self._build_base_size = box_cfg["build"]["baseSize"]
        self._build_random_size = box_cfg["build"]["randomSize"]
        self._build_random_mode_equal_proportion = box_cfg["build"]["randomModeEqualProportion"]
        self._build_x_scale_range = box_cfg["build"]["scaleRangeX"]
        self._build_y_scale_range = box_cfg["build"]["scaleRangeY"]
        self._build_z_scale_range = box_cfg["build"]["scaleRangeZ"]
        self._build_scale_sample_interval = box_cfg["build"]["scaleSampleInterval"]
        self._build_random_density = box_cfg["build"]["randomDensity"]
        self._mass_range = box_cfg["build"]["massRange"]
        self._build_test_sizes = box_cfg["build"]["testSizes"]

        self._reset_random_rot = box_cfg["reset"]["randomRot"]
        self._reset_random_height = box_cfg["reset"]["randomHeight"]
        self._reset_random_height_prob = box_cfg["reset"]["randomHeightProb"]
        self._reset_maxTopSurfaceHeight = box_cfg["reset"]["maxTopSurfaceHeight"]

        self._enable_bbox_obs = box_cfg["obs"]["enableBboxObs"]
        self.user_urdf = box_cfg["build"].get("userUrdf", None)
        self._append_mass_obs = bool(self._build_random_density)

        # Per-task settings (motion + skill distributions + object size used by task token/reward)
        default_task_settings = {
            "MMH_box": {
                "motion_file": "tokenhsi/data/dataset_carry/dataset_MMH_box.yaml",
                "skill": ["omomo", "loco", "pickUp", "carryWith", "putDown"],
                "skillInitProb": [0.0, 0.4, 0.4, 0.1, 0.1],
                "skillDiscProb": [0.0, 0.2, 0.5, 0.1, 0.1],
                "boxSize": [0.4, 0.4, 0.4],
            },
            "MMH_handle": {
                "motion_file": "tokenhsi/data/dataset_carry/dataset_MMH_handle.yaml",
                "skill": ["omomo", "loco", "pickUp", "carryWith", "putDown"],
                "skillInitProb": [0.1, 0.3, 0.6, 0.0, 0.0],
                "skillDiscProb": [0.4, 0.2, 0.4, 0.1, 0.1],
                "boxSize": [0.34, 0.34, 0.36],
            },
            "MMH_timber": {
                "motion_file": "tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml",
                "skill": ["omomo", "loco", "pickUp", "carryWith", "putDown"],
                "skillInitProb": [0.1, 0.3, 0.6, 0.0, 0.0],
                "skillDiscProb": [0.4, 0.2, 0.4, 0.1, 0.1],
                "boxSize": [0.4, 0.4, 0.4],
            },
            "MMH_bag": {
                "motion_file": "tokenhsi/data/dataset_carry/dataset_MMH_bag.yaml",
                "skill": ["omomo", "loco", "pickUp", "carryWith", "putDown"],
                "skillInitProb": [0.1, 0.3, 0.1, 0.0, 0.0],
                "skillDiscProb": [0.6, 0.2, 0.2, 0.1, 0.1],
                "boxSize": [0.35, 0.51, 0.15],
            },
        }
        self._task_settings = {}
        user_task_settings = cfg["env"].get("mmhTasks", {})
        for task_name in self._multiple_task_names:
            merged = default_task_settings[task_name].copy()
            merged.update(user_task_settings.get(task_name, {}))
            self._task_settings[task_name] = merged

        task_obs_size_single = 3 + 3 + 6 + 3 + 3 + 3 * 8 + int(self._append_mass_obs)
        self._each_subtask_obs_size = [task_obs_size_single] * self._num_tasks

        self._task_skill = {}
        self._task_skill_init_prob_raw = {}
        self._task_skill_disc_prob_raw = {}
        self._task_box_size_raw = {}
        for task_name in self._multiple_task_names:
            self._task_skill[task_name] = self._task_settings[task_name]["skill"]
            self._task_skill_init_prob_raw[task_name] = self._task_settings[task_name]["skillInitProb"]
            self._task_skill_disc_prob_raw[task_name] = self._task_settings[task_name]["skillDiscProb"]
            self._task_box_size_raw[task_name] = self._task_settings[task_name]["boxSize"]

        self._task_init_prob_raw = cfg["env"].get("taskInitProb", [1.0 / self._num_tasks] * self._num_tasks)
        self._task_disc_prob_raw = cfg["env"].get("taskDiscProb", self._task_init_prob_raw)

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidMMHMerge.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        self._task_skill_init_prob = {}
        self._task_skill_disc_prob = {}
        for task_name in self._multiple_task_names:
            if len(self._task_skill[task_name]) != len(self._task_skill_init_prob_raw[task_name]):
                raise ValueError(f"skillInitProb size mismatch for task {task_name}.")
            if len(self._task_skill[task_name]) != len(self._task_skill_disc_prob_raw[task_name]):
                raise ValueError(f"skillDiscProb size mismatch for task {task_name}.")

            task_init = torch.tensor(self._task_skill_init_prob_raw[task_name], device=self.device, dtype=torch.float)
            task_init = task_init / torch.clamp(task_init.sum(), min=1e-8)
            self._task_skill_init_prob[task_name] = task_init

            task_disc = torch.tensor(self._task_skill_disc_prob_raw[task_name], device=self.device, dtype=torch.float)
            task_disc = task_disc / torch.clamp(task_disc.sum(), min=1e-8)
            self._task_skill_disc_prob[task_name] = task_disc

        if len(self._task_init_prob_raw) != self._num_tasks:
            raise ValueError("taskInitProb size mismatch with number of MMH tasks.")
        if len(self._task_disc_prob_raw) != self._num_tasks:
            raise ValueError("taskDiscProb size mismatch with number of MMH tasks.")

        self._task_init_prob = torch.tensor(self._task_init_prob_raw, device=self.device, dtype=torch.float)
        self._task_init_prob /= torch.clamp(self._task_init_prob.sum(), min=1e-8)
        self._task_disc_prob = torch.tensor(self._task_disc_prob_raw, device=self.device, dtype=torch.float)
        self._task_disc_prob /= torch.clamp(self._task_disc_prob.sum(), min=1e-8)

        self._eval_task = cfg["args"].eval_task
        if self._is_eval and self._eval_task != "":
            if self._eval_task not in self._multiple_task_names:
                raise ValueError(f"Unsupported eval_task '{self._eval_task}'. Expected one of {self._multiple_task_names}.")
            task_id = HumanoidMMHMerge.TaskUID[self._eval_task].value
            self._task_init_prob[:] = 0.0
            self._task_disc_prob[:] = 0.0
            self._task_init_prob[task_id] = 1.0
            self._task_disc_prob[task_id] = 1.0

        # Per-task object geometry used by rewards/tokenizer obs.
        self._task_box_size = {}
        self._task_box_bps = {}
        for task_name in self._multiple_task_names:
            task_size = torch.tensor(self._task_box_size_raw[task_name], device=self.device, dtype=torch.float)
            task_size = task_size.view(1, 3).repeat(self.num_envs, 1)
            self._task_box_size[task_name] = task_size
            self._task_box_bps[task_name] = self._build_box_bps_from_size(task_size)

        self._active_task_box_size = self._task_box_size[self._multiple_task_names[0]].clone()
        self._active_task_box_bps = self._task_box_bps[self._multiple_task_names[0]].clone()

        # Optional global fallback. Each task can override motion_file through env.mmhTasks.
        motion_file = cfg["env"].get("motion_file", "")
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._task_indicator = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._task_mask = torch.zeros([self.num_envs, self._num_tasks], device=self.device, dtype=torch.bool)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._task_tar_pos = {
            task_name: torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
            for task_name in self._multiple_task_names
        }

        spacing = cfg["env"]["envSpacing"]
        if spacing <= 0.5:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-4.5, -4.5, 0.5], device=self.device),
                torch.tensor([4.5, 4.5, 1.0], device=self.device),
            )
        else:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-(spacing - 0.5), -(spacing - 0.5), 0.5], device=self.device),
                torch.tensor([(spacing - 0.5), (spacing - 0.5), 1.0], device=self.device),
            )

        if not self.headless:
            self._build_marker_state_tensors()

        # tensors for shared physical box actor
        self._build_box_tensors()

        if self._reset_random_height:
            self._build_platforms_state_tensors()

        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        if self._is_eval:
            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
            self._success_threshold = cfg["env"]["eval"]["successThreshold"]

        print(f"[Info]: constructionExp = {self.constructionExp}")
        print(f"[Info]: _is_eval = {self._is_eval}")
        print(f"[Info]: _is_test = {self._is_test}")
        if self.constructionExp and self._is_test:
            print("#" * 40, "Using construction experiment mode", "#" * 40)
            self._num_experiments = cfg["env"]["eval"].get("numExperiments", 1)
            self._fixed_start_positions = torch.tensor(cfg["env"]["eval"]["start_positions"], device=self.device, dtype=torch.float32)
            self._fixed_target_positions = torch.tensor(cfg["env"]["eval"]["end_positions"], device=self.device, dtype=torch.float32)

            assert self._fixed_start_positions.shape[0] == self._fixed_target_positions.shape[0]
            assert self._fixed_start_positions.shape[0] == self._num_experiments
            self._box_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
            self._target_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
            print(f"[Info]: _num_experiments = {self._num_experiments}")
            print(f"[Info]: _fixed_start_positions = {self._fixed_start_positions}")
            print(f"[Info]: _fixed_target_positions = {self._fixed_target_positions}")
            print(f"[Info]: _box_counter = {self._box_counter}")
            print(f"[Info]: _target_counter = {self._target_counter}")
            print(f"[Info]: _box_density_value = {self._box_density_value}")
            print(f"[Info]: init done {'#' * 40}")

        self._update_task()

    def _build_box_bps_from_size(self, size):
        bps_0 = torch.vstack([     size[:, 0] / 2,      size[:, 1] / 2, -1 * size[:, 2] / 2]).t().unsqueeze(-2)
        bps_1 = torch.vstack([-1 * size[:, 0] / 2,      size[:, 1] / 2, -1 * size[:, 2] / 2]).t().unsqueeze(-2)
        bps_2 = torch.vstack([-1 * size[:, 0] / 2, -1 * size[:, 1] / 2, -1 * size[:, 2] / 2]).t().unsqueeze(-2)
        bps_3 = torch.vstack([     size[:, 0] / 2, -1 * size[:, 1] / 2, -1 * size[:, 2] / 2]).t().unsqueeze(-2)
        bps_4 = torch.vstack([     size[:, 0] / 2,      size[:, 1] / 2,      size[:, 2] / 2]).t().unsqueeze(-2)
        bps_5 = torch.vstack([-1 * size[:, 0] / 2,      size[:, 1] / 2,      size[:, 2] / 2]).t().unsqueeze(-2)
        bps_6 = torch.vstack([-1 * size[:, 0] / 2, -1 * size[:, 1] / 2,      size[:, 2] / 2]).t().unsqueeze(-2)
        bps_7 = torch.vstack([     size[:, 0] / 2, -1 * size[:, 1] / 2,      size[:, 2] / 2]).t().unsqueeze(-2)
        return torch.cat([bps_0, bps_1, bps_2, bps_3, bps_4, bps_5, bps_6, bps_7], dim=1).to(self.device)

    def _gather_task_tensor(self, tensor_dict):
        sample = next(iter(tensor_dict.values()))
        out = torch.zeros_like(sample)
        for task_id, task_name in enumerate(self._multiple_task_names):
            mask = self._task_indicator == task_id
            if mask.sum() > 0:
                out[mask] = tensor_dict[task_name][mask]
        return out


    def get_multi_task_info(self):

        num_subtasks = self._num_tasks
        each_subtask_obs_size = self._each_subtask_obs_size

        each_subtask_obs_mask = torch.zeros(num_subtasks, sum(each_subtask_obs_size), dtype=torch.bool, device=self.device)

        index = torch.cumsum(torch.tensor([0] + each_subtask_obs_size), dim=0).to(self.device)
        for i in range(num_subtasks):
            each_subtask_obs_mask[i, index[i]:index[i + 1]] = True

        info = {
            "onehot_size": num_subtasks,
            "tota_subtask_obs_size": sum(each_subtask_obs_size),
            "each_subtask_obs_size": each_subtask_obs_size,
            "each_subtask_obs_mask": each_subtask_obs_mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": self._enable_task_mask_obs,

            "each_subtask_name": self._multiple_task_names,
        }

        return info
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        if self._reset_random_height:
            self._platform_handles = []
            self._tar_platform_handles = []
            self._load_platform_asset()

        self._box_handles = []
        self._box_masses = []
        self._load_box_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

        self._box_masses = to_torch(self._box_masses, device=self.device, dtype=torch.float32)
        return
    
    def _load_marker_asset(self):
        asset_root = "tokenhsi/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _load_platform_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        platform_size = 0.4
        # if self.constructionExp and self._is_test:
        #     platform_size = max(self._build_base_size)
        self._platform_height = 0.02
        self._platform_asset = self.gym.create_box(self.sim, platform_size, platform_size, self._platform_height, asset_options)

        return

    def _load_box_asset(self):
        
        # rescale
        self._box_scale = torch.ones((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if self._build_random_size:

            assert int((self._build_x_scale_range[1] - self._build_x_scale_range[0]) % self._build_scale_sample_interval) == 0
            assert int((self._build_y_scale_range[1] - self._build_y_scale_range[0]) % self._build_scale_sample_interval) == 0
            assert int((self._build_z_scale_range[1] - self._build_z_scale_range[0]) % self._build_scale_sample_interval) == 0

            x_scale_linespace = torch.arange(self._build_x_scale_range[0], self._build_x_scale_range[1] + self._build_scale_sample_interval, self._build_scale_sample_interval)
            y_scale_linespace = torch.arange(self._build_y_scale_range[0], self._build_y_scale_range[1] + self._build_scale_sample_interval, self._build_scale_sample_interval)
            z_scale_linespace = torch.arange(self._build_z_scale_range[0], self._build_z_scale_range[1] + self._build_scale_sample_interval, self._build_scale_sample_interval)

            if self._build_random_mode_equal_proportion == False:

                num_scales = len(x_scale_linespace) * len(y_scale_linespace) * len(z_scale_linespace)
                scale_pool = torch.zeros((num_scales, 3), device=self.device)
                idx = 0
                for curr_x in x_scale_linespace:
                    for curr_y in y_scale_linespace:
                        for curr_z in z_scale_linespace:
                            scale_pool[idx] = torch.tensor([curr_x, curr_y, curr_z])
                            idx += 1
            
            else:
                num_scales = len(x_scale_linespace)
                scale_pool = torch.zeros((num_scales, 3), device=self.device)
                idx = 0
                for curr_x in x_scale_linespace:
                    scale_pool[idx] = torch.tensor([curr_x, curr_x, curr_x])
                    idx += 1
                
                if self._mode == "test":
                    test_sizes = torch.tensor(self._build_test_sizes, device=self.device)
                    scale_pool = torch.zeros((test_sizes.shape[0], 3), device=self.device)
                    num_scales = test_sizes.shape[0]

                    for axis in range(3):
                        scale_pool[:, axis] = test_sizes[:, axis] / self._build_base_size[axis]

            if self.num_envs >= num_scales:
                self._box_scale[:num_scales] = scale_pool[:num_scales] # copy

                sampled_scale_id = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales), num_samples=(self.num_envs - num_scales), replacement=True)
                self._box_scale[num_scales:] = scale_pool[sampled_scale_id]

                shuffled_id = torch.randperm(self.num_envs)
                self._box_scale = self._box_scale[shuffled_id]

            else:
                sampled_scale_id = torch.multinomial(torch.ones(num_scales) * (1.0 / num_scales), num_samples=self.num_envs, replacement=True)
                self._box_scale = scale_pool[sampled_scale_id]

        # randomize mass
        self._box_density = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        if self._is_test and self.constructionExp:  # _box_density from the yaml file
            if self._box_density_value:
                assert self._build_random_density, "Please set build_random_density to True in the yaml file to put mass in observation"
            else:
                self._box_density_value = 100.0
            self._box_density[:] = self._box_density_value
        else:
            if self._build_random_density:
                # og code: sample by density, but may get very heavy box
                # dist = torch.distributions.uniform.Uniform(torch.tensor([80.0], device=self.device), torch.tensor([120.0], device=self.device))
                # self._box_density = dist.sample((self.num_envs,))
                
                # wen code: sample by mass
                dist = torch.distributions.uniform.Uniform(torch.tensor(self._mass_range[0], device=self.device), 
                                                           torch.tensor(self._mass_range[1], device=self.device),)
                box_mass = dist.sample((self.num_envs,))
                scale_volume = self._box_scale.prod(dim=1) 
                base_size = torch.tensor(self._build_base_size, device=self.device)
                base_volume = base_size.prod()
                box_volume = base_volume * scale_volume
                self._box_density = box_mass / box_volume
                # print("box_mass shape = ", box_mass.shape)
                # print("box_volume shape = ", box_volume.shape)
                # print("box_density shape = ", self._box_density.shape)
            else:
                self._box_density[:] = 100.0

        # print(f"[Info]: _box_density = {self._box_density}")

        self._box_size = torch.tensor(self._build_base_size, device=self.device).reshape(1, 3) * self._box_scale # (num_envs, 3)

        # create asset
        self._box_assets = []
        for i in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = self._box_density[i]
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            if self.user_urdf is not None:
                # todo: load root from userURDF
                asset_root = "tokenhsi/data/assets/carry_box"
                self._box_assets.append(self.gym.load_asset(self.sim, asset_root, f"indented_box.urdf", asset_options))
                # TODO: mass & size are taken from urdf file, not asset_options
            else: 
                self._box_assets.append(self.gym.create_box(self.sim, self._box_size[i, 0], self._box_size[i, 1], self._box_size[i, 2], asset_options))



        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_box(env_id, env_ptr)
        
        if self._reset_random_height:
            self._build_platforms(env_id, env_ptr)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = self._box_size[env_id, 0] / 2 + 0.4
        default_pose.p.y = 0
        default_pose.p.z = self._box_size[env_id, 2] / 2 # ensure no penetration between box and ground plane
    
        box_handle = self.gym.create_actor(env_ptr, self._box_assets[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)
        
        print(self.gym.get_actor_rigid_body_properties(env_ptr, box_handle)[0])
        mass = self.gym.get_actor_rigid_body_properties(env_ptr, box_handle)[0].mass
        self._box_masses.append(mass)
        print(f"[Info]: box mass = {mass} kg")
        return
    
    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs + 1
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.3)
        self._marker_handles.append(marker_handle)

        return
    
    def _build_platforms(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()

        default_pose.p.z = -5 # place under the ground
        platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.235, 0.6))

        default_pose.p.z = -5 - self._platform_height
        tar_platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "tar_platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, tar_platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))

        self._platform_handles.append(platform_handle)
        self._tar_platform_handles.append(tar_platform_handle)

        return

    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._box_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._box_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[:, 7:13] = 0

        self._build_box_bps()

        return

    def _build_box_bps(self):
        bps_0 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_1 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_2 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_3 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_4 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_5 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_6 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_7 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        self._box_bps = torch.cat([bps_0, bps_1, bps_2, bps_3, bps_4, bps_5, bps_6, bps_7], dim=1).to(self.device) # (num_envs, 8, 3)

        return

    def _build_platforms_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._platform_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._platform_pos = self._platform_states[..., :3]
        self._platform_default_pos = self._platform_pos.clone()
        self._platform_actor_ids = self._humanoid_actor_ids + 2

        self._tar_platform_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 3, :]
        self._tar_platform_pos = self._tar_platform_states[..., :3]
        self._tar_platform_default_pos = self._tar_platform_pos.clone()
        self._tar_platform_actor_ids = self._humanoid_actor_ids + 3

        return
    
    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., num_actors - 1, :]
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + (num_actors - 1)

        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()

        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size

        return obs_size

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = sum(self._each_subtask_obs_size)
            if self._enable_task_mask_obs:
                obs_size += self._num_tasks
        return obs_size
    
    def _regulate_height(self, h, box_size):
        top_surface_z = h + box_size[:, 2] / 2
        top_surface_z = torch.clamp_max(top_surface_z, self._reset_maxTopSurfaceHeight)
        return top_surface_z - box_size[:, 2] / 2

    def _update_task(self):
        self._active_task_box_size = self._gather_task_tensor(self._task_box_size)
        self._active_task_box_bps = self._gather_task_tensor(self._task_box_bps)
        self._tar_pos = self._gather_task_tensor(self._task_tar_pos)
        return

    def _reset_task(self, env_ids):
        if self._is_test and self.constructionExp:
            ids = env_ids.to(dtype=torch.long)
            target_indices = self._target_counter[ids] % self._num_experiments
            new_target_pos = self._fixed_target_positions[target_indices]
            self._tar_pos[ids] = new_target_pos
            for task_name in self._multiple_task_names:
                self._task_tar_pos[task_name][ids] = new_target_pos
            if self._reset_random_height:
                self._tar_platform_pos[ids, 0:2] = new_target_pos[:, 0:2]
                self._tar_platform_pos[ids, -1] = (
                    new_target_pos[:, -1]
                    - self._active_task_box_size[ids, 2] / 2
                    - self._platform_height / 2
                )
            self._target_counter[ids] += 1
            self._update_task()
            return

        for task_name in self._multiple_task_names:
            task_id = HumanoidMMHMerge.TaskUID[task_name].value
            task_env_ids = env_ids[self._task_indicator[env_ids] == task_id]
            if len(task_env_ids) == 0:
                continue

            task_box_size = self._task_box_size[task_name]
            self._task_tar_pos[task_name][task_env_ids] = self._box_states[task_env_ids, 0:3]

            putdown_env_ids = self._reset_ref_env_ids[task_name].get("putDown", None)
            if putdown_env_ids is not None and len(putdown_env_ids) > 0:
                motion_ids = self._reset_ref_motion_ids[task_name]["putDown"]
                motion_lib = self._motion_lib[task_name]["putDown"]
                root_pos, _ = motion_lib.get_obj_motion_state(
                    motion_ids=motion_ids,
                    motion_times=motion_lib.get_motion_length(motion_ids),
                )
                root_pos[:, 2] = task_box_size[putdown_env_ids, 2] / 2
                self._task_tar_pos[task_name][putdown_env_ids] = root_pos
                if self._reset_random_height:
                    self._tar_platform_pos[putdown_env_ids] = self._tar_platform_default_pos[putdown_env_ids]

            random_env_ids = []
            if len(self._reset_default_env_ids) > 0:
                default_mask = self._task_indicator[self._reset_default_env_ids] == task_id
                selected_default = self._reset_default_env_ids[default_mask]
                if len(selected_default) > 0:
                    random_env_ids.append(selected_default)
            for sk_name in ["loco", "pickUp", "carryWith"]:
                curr = self._reset_ref_env_ids[task_name].get(sk_name, None)
                if curr is not None and len(curr) > 0:
                    random_env_ids.append(curr)

            if len(random_env_ids) > 0:
                ids = torch.cat(random_env_ids, dim=0)
                new_target_pos = self._tar_pos_dist.sample((len(ids),))
                new_target_pos[:, 2] = task_box_size[ids, 2] / 2

                min_dist = 1.0
                target_overlap = torch.logical_or(
                    torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
                    torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist,
                )
                while torch.sum(target_overlap) > 0:
                    new_target_pos[target_overlap] = self._tar_pos_dist.sample((torch.sum(target_overlap),))
                    new_target_pos[:, 2] = task_box_size[ids, 2] / 2
                    target_overlap = torch.logical_or(
                        torch.sum((new_target_pos[..., :2] - self._humanoid_root_states[ids, :2]) ** 2, dim=-1) < min_dist,
                        torch.sum((new_target_pos[..., :2] - self._box_states[ids, :2]) ** 2, dim=-1) < min_dist,
                    )

                if self._reset_random_height:
                    num_envs = ids.shape[0]
                    probs = to_torch(np.array([self._reset_random_height_prob] * num_envs), device=self.device)
                    mask = torch.bernoulli(probs) == 1.0
                    if mask.sum() > 0:
                        new_target_pos[mask, 2] += torch.rand(mask.sum(), device=self.device) * 1.0
                        new_target_pos[mask, 2] = self._regulate_height(new_target_pos[mask, 2], task_box_size[ids[mask]])

                self._task_tar_pos[task_name][ids] = new_target_pos

                if self._reset_random_height:
                    self._tar_platform_pos[ids, 0:2] = new_target_pos[:, 0:2]
                    self._tar_platform_pos[ids, -1] = (
                        new_target_pos[:, -1]
                        - task_box_size[ids, 2] / 2
                        - self._platform_height / 2
                    )

        self._update_task()
        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return
    
    def _update_marker(self):

        self._marker_pos[:, :] = self._tar_pos[:, :]

        env_ids_int32 = torch.cat([self._marker_actor_ids, self._box_actor_ids], dim=0)
        if self._reset_random_height:
            # env has two platforms
            env_ids_int32 = torch.cat([env_ids_int32, self._platform_actor_ids, self._tar_platform_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _draw_task(self):
        self._update_task()
        self._update_marker()

        cols = np.array([
            [0.0, 1.0, 0.0], # green
            [1.0, 0.0, 0.0], # red
        ], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._box_states[..., 0:3] # line from box to marker
        ends = self._tar_pos[..., 0:3]

        starts_l2 = self._humanoid_root_states[..., 0:3] # line from humanoid to box
        ends_l2 = self._box_states[..., 0:3]

        verts = torch.cat([starts, ends, starts_l2, ends_l2], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        # draw lines of the bbox
        cols = np.zeros((24, 3), dtype=np.float32) # 24 lines
        cols[:12] = [1.0, 0.0, 0.0] # red
        cols[12:] = [0.0, 1.0, 0.0] # greed

        # transform bps from object local space to world space
        box_bps = self._box_bps.clone()
        box_pos = self._box_states[:, 0:3]
        box_rot = self._box_states[:, 3:7]
        box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
        box_bps_world_space = (quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

        verts = torch.cat([
            box_bps_world_space[:, 0, :], box_bps_world_space[:, 1, :],
            box_bps_world_space[:, 1, :], box_bps_world_space[:, 2, :],
            box_bps_world_space[:, 2, :], box_bps_world_space[:, 3, :],
            box_bps_world_space[:, 3, :], box_bps_world_space[:, 0, :],

            box_bps_world_space[:, 4, :], box_bps_world_space[:, 5, :],
            box_bps_world_space[:, 5, :], box_bps_world_space[:, 6, :],
            box_bps_world_space[:, 6, :], box_bps_world_space[:, 7, :],
            box_bps_world_space[:, 7, :], box_bps_world_space[:, 4, :],

            box_bps_world_space[:, 0, :], box_bps_world_space[:, 4, :],
            box_bps_world_space[:, 1, :], box_bps_world_space[:, 5, :],
            box_bps_world_space[:, 2, :], box_bps_world_space[:, 6, :],
            box_bps_world_space[:, 3, :], box_bps_world_space[:, 7, :],
        ], dim=-1).cpu()

        # transform bps from object local space to world space
        tar_box_pos = self._tar_pos[:, 0:3] # (num_envs, 3)
        tar_box_bps = self._active_task_box_bps
        tar_box_pos_exp = torch.broadcast_to(tar_box_pos.unsqueeze(-2), (tar_box_pos.shape[0], tar_box_bps.shape[1], tar_box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        tar_box_bps_world_space = (tar_box_bps.reshape(-1, 3) + tar_box_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

        verts_tar_box = torch.cat([
            tar_box_bps_world_space[:, 0, :], tar_box_bps_world_space[:, 1, :],
            tar_box_bps_world_space[:, 1, :], tar_box_bps_world_space[:, 2, :],
            tar_box_bps_world_space[:, 2, :], tar_box_bps_world_space[:, 3, :],
            tar_box_bps_world_space[:, 3, :], tar_box_bps_world_space[:, 0, :],

            tar_box_bps_world_space[:, 4, :], tar_box_bps_world_space[:, 5, :],
            tar_box_bps_world_space[:, 5, :], tar_box_bps_world_space[:, 6, :],
            tar_box_bps_world_space[:, 6, :], tar_box_bps_world_space[:, 7, :],
            tar_box_bps_world_space[:, 7, :], tar_box_bps_world_space[:, 4, :],

            tar_box_bps_world_space[:, 0, :], tar_box_bps_world_space[:, 4, :],
            tar_box_bps_world_space[:, 1, :], tar_box_bps_world_space[:, 5, :],
            tar_box_bps_world_space[:, 2, :], tar_box_bps_world_space[:, 6, :],
            tar_box_bps_world_space[:, 3, :], tar_box_bps_world_space[:, 7, :],
        ], dim=-1).cpu() # (num_envs, 12*6)

        bbox_verts = torch.cat([verts, verts_tar_box], dim=-1).numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = bbox_verts[i]
            curr_verts = curr_verts.reshape([24, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)
        
        radius = 0.5

        num_verts = 30
        axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        ang = torch.linspace(0, 2 * np.pi, num_verts, device=self.device)
        quat = quat_from_angle_axis(ang, axis) # (num_verts, 4)

        axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        axis = quat_rotate(quat, axis)
        pos  = axis * radius

        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        for i, env_ptr in enumerate(self.envs):
            verts = pos.clone()
            verts += self._tar_pos[i, 0:3].unsqueeze(0)
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
        
        num_verts = 30
        axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        ang = torch.linspace(0, 2 * np.pi, num_verts, device=self.device)
        quat = quat_from_angle_axis(ang, axis) # (num_verts, 4)

        axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
        axis = quat_rotate(quat, axis)
        pos  = axis * radius

        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        for i, env_ptr in enumerate(self.envs):
            verts = pos.clone()
            verts += self._box_states[i, 0:3].unsqueeze(0)
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

        return
    
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return
    
    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            box_states = self._box_states
            box_mass = self._box_masses
            task_mask = self._task_mask
        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            box_mass = self._box_masses[env_ids]
            task_mask = self._task_mask[env_ids]

        task_obs = []
        for task_name in self._multiple_task_names:
            if env_ids is None:
                box_bps = self._task_box_bps[task_name]
                tar_pos = self._task_tar_pos[task_name]
            else:
                box_bps = self._task_box_bps[task_name][env_ids]
                tar_pos = self._task_tar_pos[task_name][env_ids]

            obs = compute_location_observations(root_states, box_states, box_bps, tar_pos, self._enable_bbox_obs)
            if self._append_mass_obs:
                obs = torch.cat([obs, box_mass.unsqueeze(-1)], dim=-1)
            task_obs.append(obs)

        obs = torch.cat(task_obs, dim=-1)
        if self._enable_task_mask_obs:
            obs = torch.cat([obs, task_mask.float()], dim=-1)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        rigid_body_pos = self._rigid_body_pos
        rigid_body_rot = self._rigid_body_rot
        box_pos = self._box_states[..., 0:3]
        box_rot = self._box_states[..., 3:7]
        hands_ids = self._key_body_ids[[0, 1]]

        self._update_task()
        box_size = self._active_task_box_size
        tar_pos = self._tar_pos

        walk_r = compute_walk_reward(root_pos, self._prev_root_pos, box_pos, self.dt, 1.5,
                                     self._only_vel_reward, self.cfg["env"]["debug"]["vel"])
        carry_r = compute_carry_reward(box_pos, self._prev_box_pos, tar_pos, self.dt, 1.5, box_size,
                                       self._only_vel_reward,
                                       self._box_vel_penalty, self._box_vel_pen_coeff, self._box_vel_pen_thre,
                                       self.cfg["env"]["debug"]["vel"])
        handheld_r = compute_handheld_reward(rigid_body_pos, box_pos, hands_ids, tar_pos, self._only_height_handheld_reward)
        putdown_r = compute_putdown_reward(box_pos, tar_pos)
        carry_box_reward = walk_r + carry_r + handheld_r + putdown_r

        handle_mask = self._task_indicator == HumanoidMMHMerge.TaskUID["MMH_handle"].value
        timber_mask = self._task_indicator == HumanoidMMHMerge.TaskUID["MMH_timber"].value
        bag_mask = self._task_indicator == HumanoidMMHMerge.TaskUID["MMH_bag"].value

        if handle_mask.sum() > 0:
            handle_r = compute_handheld_handle_reward(
                rigid_body_pos[handle_mask],
                box_pos[handle_mask],
                hands_ids,
                tar_pos[handle_mask],
                self._only_height_handheld_reward,
                box_size[handle_mask],
                box_rot[handle_mask],
            )
            handheld_r[handle_mask] = handle_r
            carry_box_reward[handle_mask] = walk_r[handle_mask] + carry_r[handle_mask] + handle_r + putdown_r[handle_mask]

        if timber_mask.sum() > 0:
            timber_r = compute_handheld_timber_reward(
                rigid_body_pos[timber_mask],
                rigid_body_rot[timber_mask],
                box_pos[timber_mask],
                box_rot[timber_mask],
                box_size[timber_mask],
                hands_ids,
            )
            handheld_r[timber_mask] = timber_r
            carry_box_reward[timber_mask] = walk_r[timber_mask] + carry_r[timber_mask] + timber_r

        if bag_mask.sum() > 0:
            bag_r = compute_handheld_bag_reward(
                rigid_body_pos[bag_mask],
                rigid_body_rot[bag_mask],
                box_pos[bag_mask],
                box_rot[bag_mask],
                box_size[bag_mask],
                hands_ids,
            )
            handheld_r[bag_mask] = bag_r
            carry_box_reward[bag_mask] = walk_r[bag_mask] + carry_r[bag_mask] + bag_r

        humanoid_angles = self.humanoid_angles()
        ergo_sub_weight = torch.tensor(self._ergo_sub_weight, dtype=torch.float32)
        ergo_sub_weight /= ergo_sub_weight.sum()
        back_r = compute_back_ergo_reward(humanoid_angles["back"], humanoid_angles["left_knee"], humanoid_angles["right_knee"], ergo_sub_weight[0])
        elbow_r = compute_elbow_ergo_reward(humanoid_angles["left_elbow"], humanoid_angles["right_elbow"], rigid_body_pos, hands_ids, box_pos, self._prev_box_pos, ergo_sub_weight[1], tar_pos)
        box_r = compute_box_ergo_reward(humanoid_angles["back"], box_size, box_pos, self._prev_box_pos, rigid_body_pos, hands_ids, ergo_sub_weight[2], tar_pos)
        ergo_reward = back_r + elbow_r + box_r
        total_reward = carry_box_reward * (1 - self._ergo_coeff) + ergo_reward * self._ergo_coeff
        
        if self._verbose:
            print("#"*40)
            print(f"""[Info] Frame {self.frame_count}
                Ergo coeff = {self._ergo_coeff}
                Carry_box_reward = {carry_box_reward* (1 - self._ergo_coeff)}
                - walk_r       = {walk_r* (1 - self._ergo_coeff)}
                - carry_r      = {carry_r* (1 - self._ergo_coeff)}
                - handheld_r   = {handheld_r* (1 - self._ergo_coeff)}
                - putdown_r    = {putdown_r* (1 - self._ergo_coeff)}
                Ergo_reward  = {ergo_reward * self._ergo_coeff}
                - back_r       = {back_r * self._ergo_coeff}
                - elbow_r      = {elbow_r * self._ergo_coeff}
                - box_r        = {box_r * self._ergo_coeff}
                """)
            self.print_angles_degrees(humanoid_angles)
            # print(rigid_body_pos[0])
            # print(f"hand_pos = {rigid_body_pos[0][hands_ids]}")
            # print(f"box_pos = {box_pos[0]}")
        # only show 0th env reward
        metrics = {
            "reward/CARRY_BOX": carry_box_reward[0].item(),
            "reward/walk_r": walk_r[0].item(),
            "reward/carry_r": carry_r[0].item(),
            "reward/handheld_r": handheld_r[0].item(),
            "reward/putdown_r": putdown_r[0].item(),
            "reward_ergo/ERGO": ergo_reward[0].item(),
            "reward_ergo/back_r": back_r[0].item(),
            "reward_ergo/elbow_r": elbow_r[0].item(),
            "reward_ergo/box_r": box_r[0].item(),
            "reward/frames": self.frame_count,
            "reward/total_reward": total_reward[0].item(),
            }
        wandb.log(metrics, step=self.frame_count)
        
        metrics["ergo_coeff"] = self._ergo_coeff
        reward_file = f"{self.save_video_dir}/rewards.csv"
        if (self.viewer and self.save_video) or (self.headless and self.record_headless):
            if np.mod(self.frame_count, self.downsample) == 0:
                # metrics dict value to 
                csv_row = metrics.values()
                self.write_csv_row(reward_file, csv_row, header=metrics.keys())
                 
        if (box_r[0]).item()>0:
            pass  # place to put breakpoint to check reward
        carry_box_reward = total_reward + 0.0
        
        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self._power_coefficient * power

        if self._power_reward:
            self.rew_buf[:] = carry_box_reward + power_reward
        else:
            self.rew_buf[:] = carry_box_reward

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights,)
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return
    
    def _reset_boxes(self, env_ids):
        self._update_task()
        if self._is_test and self.constructionExp:  # wen: specify box location from yaml instead of random
            ids = env_ids.to(dtype=torch.long)
            # Use the same or a separate experiment counter as needed (here we assume the same)
            if self._box_counter[ids] >= 45:
                raise ValueError(f"Intential: breaking loop for {self._box_counter[ids]} boxes")
            start_indices = self._box_counter[ids] % self._fixed_start_positions.shape[0]
            root_pos = self._fixed_start_positions[start_indices]
            self._box_states[ids, 0:3] = root_pos

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
            # if self._reset_random_rot:
            #     coeff = 1.0
            # else:
            coeff = 0.0 # lets just set to 0 for now
            ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi * coeff
            root_rot = quat_from_angle_axis(ang, axis)

            self._box_states[ids, 3:7] = root_rot
            self._box_states[ids, 7:10] = 0.0
            self._box_states[ids, 10:13] = 0.0


            self._platform_pos[ids, 0:2] = root_pos[:, 0:2] # xy
            self._platform_pos[ids, -1] = root_pos[:, -1] - self._active_task_box_size[ids, 2] / 2 - self._platform_height / 2

            self._box_states[ids, 2] += 0.05 # add 0.05 to enable correct collision detection
            

            print(f"[Info]: _box_counter = {self._box_counter[ids]}")
            print(f"[Info]: _reset_boxes: box_pos = {self._box_states[ids, 0:3]}")
            print(f"[Info]: box reset {'#'*40}")
            # Increment the counter here if you want start and target to rotate together.
            self._box_counter[ids] += 1
            return
        # For pickUp/carryWith/putDown: use object state from corresponding reference motion.
        for task_name in self._multiple_task_names:
            task_box_size = self._task_box_size[task_name]
            for sk_name in ["pickUp", "carryWith", "putDown"]:
                curr_env_ids = self._reset_ref_env_ids[task_name].get(sk_name, None)
                if curr_env_ids is None or len(curr_env_ids) == 0:
                    continue

                root_pos, root_rot = self._motion_lib[task_name][sk_name].get_obj_motion_state(
                    motion_ids=self._reset_ref_motion_ids[task_name][sk_name],
                    motion_times=self._reset_ref_motion_times[task_name][sk_name],
                )

                on_ground_mask = task_box_size[curr_env_ids, 2] / 2 > root_pos[:, 2]
                root_pos[on_ground_mask, 2] = task_box_size[curr_env_ids[on_ground_mask], 2] / 2

                self._box_states[curr_env_ids, 0:3] = root_pos
                self._box_states[curr_env_ids, 3:7] = root_rot
                self._box_states[curr_env_ids, 7:10] = 0.0
                self._box_states[curr_env_ids, 10:13] = 0.0

                if self._reset_random_height:
                    self._platform_pos[curr_env_ids] = self._platform_default_pos[curr_env_ids]

        # For loco or default-reset envs: randomize initial box state.
        for task_name in self._multiple_task_names:
            task_id = HumanoidMMHMerge.TaskUID[task_name].value
            task_box_size = self._task_box_size[task_name]

            random_env_ids = []
            if len(self._reset_default_env_ids) > 0:
                default_mask = self._task_indicator[self._reset_default_env_ids] == task_id
                selected_default = self._reset_default_env_ids[default_mask]
                if len(selected_default) > 0:
                    random_env_ids.append(selected_default)

            loco_ids = self._reset_ref_env_ids[task_name].get("loco", None)
            if loco_ids is not None and len(loco_ids) > 0:
                random_env_ids.append(loco_ids)

            if len(random_env_ids) == 0:
                continue

            ids = torch.cat(random_env_ids, dim=0)
            root_pos_xy = torch.randn(len(ids), 2, device=self.device)
            root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
            root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 9.0 + 1.0
            root_pos_xy += self._humanoid_root_states[ids, :2]

            root_pos_z = task_box_size[ids, 2] / 2
            if self._reset_random_height:
                num_envs = ids.shape[0]
                probs = to_torch(np.array([self._reset_random_height_prob] * num_envs), device=self.device)
                mask = torch.bernoulli(probs) == 1.0
                if mask.sum() > 0:
                    root_pos_z[mask] += torch.rand(mask.sum(), device=self.device) * 1.0
                    root_pos_z[mask] = self._regulate_height(root_pos_z[mask], task_box_size[ids[mask]])

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
            coeff = 1.0 if self._reset_random_rot else 0.0
            ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi * coeff
            root_rot = quat_from_angle_axis(ang, axis)
            root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

            self._box_states[ids, 0:3] = root_pos
            self._box_states[ids, 3:7] = root_rot
            self._box_states[ids, 7:10] = 0.0
            self._box_states[ids, 10:13] = 0.0

            if self._reset_random_height:
                self._platform_pos[ids, 0:2] = root_pos[:, 0:2]
                self._platform_pos[ids, -1] = root_pos[:, -1] - task_box_size[ids, 2] / 2 - self._platform_height / 2
                self._box_states[ids, 2] += 0.05 # add 0.05 to enable correct collision detection

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')

        env_ids_int32 = self._box_actor_ids[env_ids].view(-1)
        if self._reset_random_height:
            # env has two platforms
            env_ids_int32 = torch.cat([env_ids_int32, self._platform_actor_ids, self._tar_platform_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["policy_obs"] = self.obs_buf.clone()

        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf

        return

    def _compute_metrics_evaluation(self):
        box_root_pos = self._box_states[..., 0:3]

        pos_diff = self._tar_pos - box_root_pos
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1

        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask], pos_err[dist_mask], self._precision_buf[dist_mask])

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        task_id = torch.multinomial(self._task_disc_prob, num_samples=1, replacement=True).item()
        task_name = self._multiple_task_names[task_id]

        skill_prob = self._task_skill_disc_prob[task_name]
        sk_id = torch.multinomial(skill_prob, num_samples=1, replacement=True).item()
        sk_name = self._task_skill[task_name][sk_id]
        curr_motion_lib = self._motion_lib[task_name][sk_name]

        task_onehot = torch.zeros(num_samples * self._num_amp_obs_steps, self._num_tasks, device=self.device, dtype=torch.float32)
        task_onehot[:, task_id] = 1.0

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = curr_motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = curr_motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0, curr_motion_lib)

        if self._enable_task_specific_disc:
            amp_obs_demo = torch.cat([amp_obs_demo, task_onehot], dim=-1)

        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml") or (asset_file == "mjcf/phys_humanoid_v3_box_foot.xml") or ("mjcf/phys_humanoid_v3_box_foot_tall" in asset_file):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif ("mjcf/phys_humanoid_v4" in asset_file or "mjcf/phys_humanoid_v5" in asset_file):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 2 * 3 + 3 * num_key_bodies
        else:
            print("Unsupported character config file: {s}".format(asset_file=asset_file))
            assert(False)

        if self._enable_task_specific_disc:
            self._num_amp_obs_per_step += self._num_tasks

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)

        self._motion_lib = {}
        for task_name in self._multiple_task_names:
            task_motion_file = self._task_settings[task_name].get("motion_file", motion_file)
            if task_motion_file is None or task_motion_file == "":
                raise ValueError(f"Missing motion_file for task {task_name}.")

            ext = os.path.splitext(task_motion_file)[1]
            if ext != ".yaml":
                raise NotImplementedError

            with open(os.path.join(os.getcwd(), task_motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            self._motion_lib[task_name] = {}
            motion_skills = set(motion_config["motions"].keys())
            for skill in self._task_skill[task_name]:
                if skill not in motion_skills:
                    raise KeyError(f"Skill '{skill}' is not in {task_motion_file} for task {task_name}.")
                self._motion_lib[task_name][skill] = MotionLib(
                    motion_file=task_motion_file,
                    skill=skill,
                    dof_body_ids=self._dof_body_ids,
                    dof_offsets=self._dof_offsets,
                    key_body_ids=self._key_body_ids.cpu().numpy(),
                    device=self.device,
                )

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_task_indicator(env_ids)
            self._reset_actors(env_ids)
            self._reset_boxes(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidMMHMerge.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidMMHMerge.StateInit.Start
              or self._state_init == HumanoidMMHMerge.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidMMHMerge.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_task_indicator(self, env_ids):
        self._task_indicator[env_ids] = torch.multinomial(self._task_init_prob, num_samples=len(env_ids), replacement=True)
        self._task_mask[env_ids] = False
        for task_id in range(self._num_tasks):
            task_env_ids = env_ids[self._task_indicator[env_ids] == task_id]
            if len(task_env_ids) > 0:
                self._task_mask[task_env_ids, task_id] = True

        for task_name in self._multiple_task_names:
            self._reset_ref_env_ids[task_name] = {}
            self._reset_ref_motion_ids[task_name] = {}
            self._reset_ref_motion_times[task_name] = {}

        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

        self._kinematic_humanoid_rigid_body_states[env_ids] = self._initial_humanoid_rigid_body_states[env_ids]

        self._every_env_init_dof_pos[env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _reset_ref_state_init(self, env_ids):
        for task_name in self._multiple_task_names:
            task_id = HumanoidMMHMerge.TaskUID[task_name].value
            task_env_ids = env_ids[self._task_indicator[env_ids] == task_id]
            if len(task_env_ids) == 0:
                continue

            skill_list = self._task_skill[task_name]
            skill_init_prob = self._task_skill_init_prob[task_name]
            sampled_skill_ids = torch.multinomial(skill_init_prob, num_samples=len(task_env_ids), replacement=True)

            for uid, sk_name in enumerate(skill_list):
                curr_env_ids = task_env_ids[sampled_skill_ids == uid]
                if len(curr_env_ids) == 0:
                    continue

                curr_motion_lib = self._motion_lib[task_name][sk_name]
                motion_ids = curr_motion_lib.sample_motions(len(curr_env_ids))

                if self._state_init == HumanoidMMHMerge.StateInit.Start:
                    motion_times = torch.zeros(len(curr_env_ids), device=self.device)
                elif (
                    self._state_init == HumanoidMMHMerge.StateInit.Random
                    or self._state_init == HumanoidMMHMerge.StateInit.Hybrid
                ):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids)
                else:
                    assert False, f"Unsupported state initialization strategy: {self._state_init}"

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _ = curr_motion_lib.get_motion_state(motion_ids, motion_times)

                self._set_env_state(
                    env_ids=curr_env_ids,
                    root_pos=root_pos,
                    root_rot=root_rot,
                    dof_pos=dof_pos,
                    root_vel=root_vel,
                    root_ang_vel=root_ang_vel,
                    dof_vel=dof_vel,
                )
                self._humanoid_root_states[curr_env_ids, 2] += 0.1

                body_pos, body_rot, body_vel, body_ang_vel = curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
                self._kinematic_humanoid_rigid_body_states[curr_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
                self._every_env_init_dof_pos[curr_env_ids] = dof_pos

                self._reset_ref_env_ids[task_name][sk_name] = curr_env_ids
                self._reset_ref_motion_ids[task_name][sk_name] = motion_ids
                self._reset_ref_motion_times[task_name][sk_name] = motion_times

        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        for task_name in self._multiple_task_names:
            for sk_name in self._task_skill[task_name]:
                curr_env_ids = self._reset_ref_env_ids[task_name].get(sk_name, None)
                if curr_env_ids is not None and len(curr_env_ids) > 0:
                    self._init_amp_obs_ref(
                        curr_env_ids,
                        self._reset_ref_motion_ids[task_name][sk_name],
                        self._reset_ref_motion_times[task_name][sk_name],
                        task_name,
                        sk_name,
                    )

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times, task_name, skill_name):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib[task_name][skill_name].get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        

        if self._enable_task_specific_disc:
            task_id = HumanoidMMHMerge.TaskUID[task_name].value
            motion_labels = torch.zeros((env_ids.shape[0], self._num_tasks), device=self.device, dtype=torch.bool)
            motion_labels[:, task_id] = True
            motion_labels = torch.broadcast_to(motion_labels.unsqueeze(-2), [motion_labels.shape[0], self._num_amp_obs_steps - 1, motion_labels.shape[1]])

            amp_obs_demo = torch.cat([amp_obs_demo, motion_labels.reshape(-1, motion_labels.shape[-1]).float()], dim=-1)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            amp_obs = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
            
            if self._enable_task_specific_disc:
                self._curr_amp_obs_buf[:] = torch.cat([amp_obs, self._task_mask.float()], dim=-1)
            else:
                self._curr_amp_obs_buf[:] = amp_obs

        else:
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            amp_obs = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
            if self._enable_task_specific_disc:
                self._curr_amp_obs_buf[env_ids] = torch.cat([amp_obs, self._task_mask[env_ids].float()], dim=-1)
            else:
                self._curr_amp_obs_buf[env_ids] = amp_obs
        return
