#!/bin/bash -l
#SBATCH --job-name=TokenHSI-MMHstage1-train
#SBATCH --output=output_slurm/train_MMHstage1_log.txt
#SBATCH --error=output_slurm/train_MMHstage1_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
##### END preamble
##### Run in MotionBert dir

my_job_header

echo "=== NVIDIA SMI ==="
nvidia-smi
# 80 hr

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

python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_construction.yaml \
    --num_envs 10240 \
    --headless \
    --wandb_project "TokenHSI-MMH-Train" \
    --wandb_mode "online" \
    --box_w 0.4 \
    --random_size True \
    --random_density True \
    --random_mode_equal_proportion True \
    --construction_experiment False \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try1/Carry-box-train-1/ \
    --wandb_name "Carry-box-train-1" \
    --notes "box w. 1 good motion and 0.2 reward" \
    --ergo_coeff 0.2 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try4/Carry-GoodMotion-resume-ergoReward-train-9/Humanoid_19-12-53-48/nn/Humanoid.pth \

