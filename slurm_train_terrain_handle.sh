#!/bin/bash -l
#SBATCH --job-name=TokenHSI-train-terrain-handle
#SBATCH --output=output_slurm/train_terrain_handle_log.txt
#SBATCH --error=output_slurm/train_terrain_handle_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --account=shdpm98
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

echo "=== vulkaninfo ==="
vulkaninfo | grep -i "version" | grep -i "vulkan" | head -n 1

echo ""

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"

# export MAX_JOBS=1


# box try
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_handle.yaml \
    --hrl_checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/Stage1/Stage1-GoodMotion-scratch-train-1/Humanoid_14-22-34-39/nn/Humanoid.pth \
    --num_envs 2048 \
    --headless \
    --wandb_project "TokenHSI-Train" \
    --wandb_mode "online" \
    --random_density False \
    --random_mode_equal_proportion False \
    --construction_experiment False \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_handle.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Terrain-handle/Terrain-handle-boxContinue-3/ \
    --wandb_name "Terrain-handle-boxContinue-3" \
    --notes "new reward" \
    --ergo_coeff 0.2 \
    --unwalkable_obstacles 0 \
    --resume 1 \
    --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Terrain-box/Terrain-box-scratch-1/Humanoid_21-16-40-13/nn/Humanoid.pth \
# --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Terrain-handle/Terrain-handle-scratch-1/Humanoid_17-00-37-28/nn/Humanoid.pth \
