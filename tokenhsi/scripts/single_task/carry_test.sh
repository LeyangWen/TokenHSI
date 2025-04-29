#!/bin/bash

python ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint output/single_task/ckpt_carry.pth \
    --test \
    --num_envs 16



python ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint output/custom_trained/density/Humanoid_1-7.pth \
    --test \
    --random_density True \
    --num_envs 1