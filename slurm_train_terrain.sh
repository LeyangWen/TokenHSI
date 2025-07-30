#!/bin/bash -l
#SBATCH --job-name=TokenHSI-train
#SBATCH --output=output_slurm/train_terrain_log.txt
#SBATCH --error=output_slurm/train_terrain_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
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

python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --num_envs 2048 \
    --headless \
    --wandb_project "TokenHSI-Train" \
    --wandb_mode "online" \
    --box_w 0.4 \
    --random_size True \
    --random_density False \
    --random_mode_equal_proportion True \
    --construction_experiment False \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try6/Terrain-GoodMotion-pretrainStage1-train-1/ \
    --wandb_name "Try6-Terrain-GoodMotion-pretrainStage1-train-1" \
    --notes "motion only" \
    --ergo_coeff 0.0 \

    # --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try6/Terrain-Reward-pretrainStage1-train-3/ \
    # --wandb_name "Try6-Terrain-Reward-pretrainStage1-train-3" \
    # --notes "reward only" \
    # --ergo_coeff 0.2 \

    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try6/Terrain-GoodMotion-Reward-pretrainStage1-train-2/ \
    # --wandb_name "Try6-Terrain-GoodMotion-Reward-pretrainStage1-train-2" \
    # --notes "good motion + reward" \
    # --ergo_coeff 0.2 \

    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try6/Terrain-GoodMotion-pretrainStage1-train-1/ \
    # --wandb_name "Try6-Terrain-GoodMotion-pretrainStage1-train-1" \
    # --notes "motion only" \
    # --ergo_coeff 0.0 \



    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try5/Terrain-GoodMotion-resume-ergoReward-train-3/ \
    # --wandb_name "Terrain-GoodMotion-resume-ergoReward-train-3" \
    # --notes "good motion only, then ergo reward (debug: added ergo to total), resume on exp1 (Humanoid_22-18-43-04)" \
    # --ergo_coeff 0.2 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try5/Terrain-GoodMotion-pretrainStage1-train-1/Humanoid_22-18-43-04/nn/Humanoid.pth \


    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try5/Terrain-GoodMotion-resume-ergoReward-train-3/ \
    # --wandb_name "Terrain-GoodMotion-resume-ergoReward-train-3" \
    # --notes "good motion only, then ergo reward (debug: added ergo to total), resume on exp1 (Humanoid_22-18-43-04)" \
    # --ergo_coeff 0.2 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try5/Terrain-GoodMotion-pretrainStage1-train-1/Humanoid_22-18-43-04/nn/Humanoid.pth \






    # 0.01 0.2 0.4 0.6 0.8 0.99
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
