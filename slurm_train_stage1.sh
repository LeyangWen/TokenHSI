#!/bin/bash -l
#SBATCH --job-name=TokenHSI-stage1-train
#SBATCH --output=output_slurm/train_stage1_log.txt
#SBATCH --error=output_slurm/train_stage1_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
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

echo "=== vulkaninfo ==="
vulkaninfo | grep -i "version" | grep -i "vulkan" | head -n 1

echo ""

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"

# export MAX_JOBS=1
python ./tokenhsi/run.py --task HumanoidMMHMerge \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_MMH.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_MMH_merge_construction.yaml \
    --num_envs 4096 \
    --headless \
    --wandb_project "TokenHSI-Train" \
    --wandb_mode "online" \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/Stage1/Stage1-GoodMotion-scratch-train-1/ \
    --wandb_name "Try8-Stage1-GoodMotion-scratch-train-1" \
    --notes "motion only, v5 humnaoid" \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_12-15-07-56/nn/Humanoid.pth \
