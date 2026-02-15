#!/bin/bash -l
#SBATCH --job-name=TokenHSI-MMH-train
#SBATCH --output=output_slurm/train_MMH_log_bag.txt
#SBATCH --error=output_slurm/train_MMH_error_bag.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --account=shdpm98
#SBATCH --partition=spgpu
##### END preamble

my_job_header

echo "=== NVIDIA SMI ==="
nvidia-smi
# 80 hr

conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch

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


echo "=== NVCC Version ==="
nvcc --version

# export MAX_JOBS=1

# box w. bag
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_bag_construction.yaml \
    --num_envs 10240 \
    --headless \
    --wandb_project "TokenHSI-MMH-Train" \
    --wandb_mode "online" \
    --random_mode_equal_proportion False \
    --construction_experiment False \
    --random_density True \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try4/Carry-bag-train-3/ \
    --wandb_name "Try4-Carry-bag-train-3" \
    --notes "Scratch, imitation motion and reward" \
    --ergo_coeff 0.2 \
    # --resume 1\
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try4/Carry-bag-train-1/Humanoid_03-22-51-45/nn/Humanoid.pth
