#!/bin/bash -l
#SBATCH --job-name=TokenHSI-test
#SBATCH --output=output_slurm/log.txt
#SBATCH --error=output_slurm/error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=5:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
##### END preamble
##### Run in MotionBert dir

my_job_header

echo "=== NVIDIA SMI ==="
nvidia-smi


conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch


echo "=== NVCC Version ==="
nvcc --version

echo "=== Python Version ==="
python --version

echo "=== PyTorch Version ==="
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"

# export MAX_JOBS=1

> joint_states.csv

# # sh tokenhsi/scripts/single_task/carry_test.sh
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint output/single_task/ckpt_carry.pth \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/carry1/ \
    --test \
    --num_envs 1 \
    --headless \
    --record_headless \

# sh tokenhsi/scripts/tokenhsi/stage2_terrain_carry_test.sh
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint output/tokenhsi/ckpt_stage2_terrainShape_carry.pth \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/carry_terrain1/ \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "CarryTerrain_test" \
    --notes "rand loc, test carry" \
    --headless \
    --record_headless \

# python lpanlib/others/video.py --imgs_dir output/imgs/example_path --delete_imgs

    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/train_exp_1/Humanoid_09-00-42-07/nn/Humanoid.pth \
#     --box_w 1.0 \
    # --box_h 1.5 \
    # --box_l 2.0 \
    # --random_size False \
    # --random_mode_equal_proportion True \
    # --scale_sample_interval 0.1 \
    # --random_density False \
    # --num_experiments 5 \
    # --construction_experiment False \
    # --start_positions "[[-3.0, -3.0, 1.0], [3.0, -3.0, 0.8]]" \
    # --end_positions "[[0.0, 0.0, 0.5], [0.0, 0.0, 0.75]]" \
    # --density 120.0 \