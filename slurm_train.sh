#!/bin/bash -l
#SBATCH --job-name=TokenHSI-train
#SBATCH --output=output_slurm/train_log_1.txt
#SBATCH --error=output_slurm/train_error_1.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
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

python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --num_envs 4096 \
    --headless \
    --wandb_project "TokenHSI-Train" \
    --wandb_mode "online" \
    --box_w 0.4 \
    --random_size True \
    --random_density True \
    --random_mode_equal_proportion True \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try2/Carry-NewMotion-train-1/ \
    --wandb_name "Try2-Carry-NewMotion-train-1-resume" \
    --notes "RndMass [2,25], 0.4 size, resume" \
    --resume 1 \
    --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try2/Carry-NewMotion-train-1/Humanoid_07-04-13-24/nn/Humanoid.pth \
    --ergo_coeff 0.2 \
    
    
    #     --box_w 1.0 \
    # --box_h 1.5 \
    # --box_l 2.0 \
    # --random_size False \

    # --scale_sample_interval 0.1 \
    # --random_density False \
    # --num_experiments 5 \
    # --construction_experiment False \
    # --start_positions "[[-3.0, -3.0, 1.0], [3.0, -3.0, 0.8]]" \
    # --end_positions "[[0.0, 0.0, 0.5], [0.0, 0.0, 0.75]]" \
    # --density 120.0 \
