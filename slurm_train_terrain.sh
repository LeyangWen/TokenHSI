#!/bin/bash -l
#SBATCH --job-name=TokenHSI-train-terrain
#SBATCH --output=output_slurm/train_terrain_log.txt
#SBATCH --error=output_slurm/train_terrain_error.txt
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
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --hrl_checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/Stage1/Stage1-GoodMotion-scratch-train-1/Humanoid_14-22-34-39/nn/Humanoid.pth \
    --num_envs 2048 \
    --headless \
    --wandb_project "TokenHSI-Train" \
    --wandb_mode "online" \
    --box_w 0.4 \
    --random_size True \
    --random_density False \
    --random_mode_equal_proportion True \
    --construction_experiment False \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Terrain-box/Terrain-box-scratch-1/ \
    --wandb_name "Terrain-box-scratch-1" \
    --notes "new humanoid" \
    --ergo_coeff 0.2 \
    --unwalkable_obstacles 0 \
    --resume 1 \
    --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Terrain-box/Terrain-box-scratch-1/Humanoid_14-23-10-07/nn/Humanoid.pth \



# python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
#     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
#     --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_slope.yaml \
#     --hrl_checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \
#     --num_envs 2048 \
#     --headless \
#     --wandb_project "TokenHSI-Train" \
#     --wandb_mode "online" \
#     --box_w 0.4 \
#     --random_size True \
#     --random_density False \
#     --random_mode_equal_proportion True \
#     --construction_experiment False \
#     --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
#     --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/Terrain-GoodMotion-Reward-scratch-4/ \
#     --wandb_name "Try9-Terrain-GoodMotion-Reward-scratch-4" \
#     --notes "reward bug fix based on Try9-2" \
#     --ergo_coeff 0.2 \
#     --unwalkable_obstacles 0 \
#     --resume 1 \
#     --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/Terrain-GoodMotion-Reward-scratch-2/Humanoid_23-03-56-14/nn/Humanoid.pth \


### 3 exp resume from scratch
    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/TerrainObstacles-GoodMotion-Reward-scratch-1/ \
    # --wandb_name "Try9-TerrainObstacles-GoodMotion-Reward-scratch-1" \
    # --notes "good motion reward, resume, obstacles, from scratch" \
    # --ergo_coeff 0.2 \
    # --unwalkable_obstacles 60 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/TerrainObstacles-GoodMotion-Reward-scratch-1/Humanoid_19-12-46-02/nn/Humanoid.pth \


    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/Terrain-GoodMotion-Reward-scratch-2/ \
    # --wandb_name "Try9-Terrain-GoodMotion-Reward-scratch-2" \
    # --notes "good motion reward, resume, from scratch" \
    # --ergo_coeff 0.2 \
    # --unwalkable_obstacles 0 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/Terrain-GoodMotion-Reward-scratch-2/Humanoid_19-12-50-16/nn/Humanoid.pth \

    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/Terrain-GoodMotion-scratch-3/ \
    # --wandb_name "Try9-Terrain-GoodMotion-resume-scratch-3" \
    # --notes "good motion, resume, from scratch" \
    # --ergo_coeff 0.0 \
    # --unwalkable_obstacles 0 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try9/Terrain-GoodMotion-scratch-3/Humanoid_19-12-50-39/nn/Humanoid.pth \













### 3 exp resume from pretrain
    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/TerrainObstacles-GoodMotion-Reward-resume-pretrained-5/ \
    # --wandb_name "Try8-TerrainObstacles-GoodMotion-Reward-pretrained-5" \
    # --notes "good motion reward, resume, obstacles, resume on pretrained tokenhsi stage2 terrain carry" \
    # --ergo_coeff 0.2 \
    # --unwalkable_obstacles 60 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/TerrainObstacles-GoodMotion-Reward-resume-pretrained-5/Humanoid_19-12-43-50/nn/Humanoid.pth \

    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-Reward-resume-pretrained-6/ \
    # --wandb_name "Try8-Terrain-GoodMotion-Reward-pretrained-6" \
    # --notes "good motion reward, resume, obstacles, resume on pretrained tokenhsi stage2 terrain carry" \
    # --ergo_coeff 0.2 \
    # --unwalkable_obstacles 0 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-Reward-resume-pretrained-6/Humanoid_12-17-31-06/nn/Humanoid.pth \
    
    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-resume-pretrained-7/ \
    # --wandb_name "Try8-Terrain-GoodMotion-pretrained-7" \
    # --notes "good motion, resume, resume on pretrained tokenhsi stage2 terrain carry" \
    # --ergo_coeff 0.0 \
    # --resume 1 \
    # --unwalkable_obstacles 0 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-resume-pretrained-7/Humanoid_12-17-28-41/nn/Humanoid.pth




    # --checkpoint output/tokenhsi/ckpt_stage2_terrainShape_carry.pth \  # tokenhsi stage2 terrain carry pretrained

    # --checkpoint output/custom_trained/try8/TerrainObstacles-GoodMotion-Reward-resume-pretrained-5/Humanoid_08-15-55-51/nn/Humanoid.pth # motion+reward + obstacles (very good reward)
# --checkpoint output/custom_trained/try8/Terrain-GoodMotion-resume-pretrained-7/Humanoid_09-06-20-48/nn/Humanoid.pth # motion (okay reward)
# --checkpoint output/custom_trained/try8/Terrain-GoodMotion-Reward-resume-pretrained-6/Humanoid_08-15-59-52/nn/Humanoid.pth # motion+reward




    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/TerrainObstacles-GoodMotion-Reward-resume-8-1-train-4/ \
    # --wandb_name "Try8-TerrainObstacles-GoodMotion-Reward-resume-8-1-train-4 (spgpu)" \
    # --notes "good motion, resume, obstacles, resume on try8-4" \
    # --ergo_coeff 0.2 \
    # --unwalkable_obstacles 30 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/TerrainObstacles-GoodMotion-Reward-resume-8-1-train-4/Humanoid_04-11-32-47/nn/Humanoid.pth \


    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-Reward-resume-8-1-train-2/ \
    # --wandb_name "Try8-Terrain-GoodMotion-resume-8-1-train-3" \
    # --notes "good motion, resume on try8-3" \
    # --ergo_coeff 0.2 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-Reward-resume-8-1-train-2/Humanoid_04-11-31-15/nn/Humanoid.pth \





    # --construction_experiment False \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-resume-8-1-train-3/ \
    # --wandb_name "Try8-Terrain-GoodMotion-Reward-resume-8-1-train-2" \
    # --notes "good motion reward, resume on try8-2" \
    # --ergo_coeff 0.0 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/try8/Terrain-GoodMotion-resume-8-1-train-3/Humanoid_04-11-29-42/nn/Humanoid.pth \





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
