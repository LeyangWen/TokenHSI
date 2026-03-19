#!/bin/bash

target_task=$1 # MMH_box, MMH_handle, MMH_timber, MMH_bag

python ./tokenhsi/run.py --task HumanoidMMHMerge \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_MMH.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_MMH_merge_construction.yaml \
    --checkpoint output/tokenhsi/ckpt_stage1.pth \
    --test \
    --num_envs 512 \
    --headless \
    --eval \
    --eval_task $target_task \
    --seed 0
