#!/bin/bash

python ./tokenhsi/run.py --task HumanoidMMHMerge \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_MMH.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_MMH_merge_construction.yaml \
    --num_envs 4096 \
    --headless
