#!/bin/bash -l
#SBATCH --job-name=TokenHSI-MMH-train
#SBATCH --output=output_slurm/train_MMH_log.txt
#SBATCH --error=output_slurm/train_MMH_error.txt
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
# python -u ./tokenhsi/run.py --task HumanoidCarryMMH \
    # --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \


# # box, v3 for now, update to v4 todo
# python -u ./tokenhsi/run.py --task HumanoidCarry \
#     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
#     --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_construction.yaml \
#     --num_envs 10240 \
#     --headless \
#     --wandb_project "TokenHSI-MMH-Train" \
#     --wandb_mode "online" \
#     --random_size True \
#     --random_density True \
#     --random_mode_equal_proportion False \
#     --construction_experiment False \
#     --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
#     --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try1/Carry-box-train-5-wrist/ \
#     --wandb_name "Carry-box-train-5-wrist" \
#     --notes "added movable wrist, 43 kg max mass, added high motion" \
#     --ergo_coeff 0.2 \
#     # --resume 1 \
#     # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try1/Carry-box-train-5-wrist/Humanoid_28-03-16-52/nn/Humanoid.pth

#     # --wandb_mode "disabled" \


# timber v4
# python -u ./tokenhsi/run.py --task HumanoidCarry \
#     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
#     --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction.yaml \
#     --num_envs 10240 \
#     --headless \
#     --wandb_project "TokenHSI-MMH-Train" \
#     --wandb_mode "online" \
#     --random_size False \
#     --random_density True \
#     --box_w 0.095 \
#     --box_l 1.8 \
#     --box_h 0.045 \
#     --random_mode_equal_proportion False \
#     --construction_experiment False \
#     --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
#     --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try1/Carry-timber-train-17/ \
#     --wandb_name "Carry-timber-train-17" \
#     --notes "resume horz-hand; more omomo type motions" \
#     --ergo_coeff 0.2 \
#     --resume 1 \
#     --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try1/Carry-timber-train-14/Humanoid_06-05-02-09/nn/Humanoid.pth


# python -u ./tokenhsi/run.py --task HumanoidCarry \
#     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
#     --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction.yaml \
#     --num_envs 10240 --headless \
#     --wandb_project TokenHSI-MMH-Train \
#     --wandb_mode online --random_size False --random_density True \
#     --box_w 0.095 --box_l 1.8 --box_h 0.045 \
#     --random_mode_equal_proportion False \
#     --construction_experiment False \
#     --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
#     --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try2/Carry-timber-train-1/ \
#     --wandb_name Try2-Carry-timber-train-1 \
#     --notes "resume try1-19 good results, 60s, timber reward, longer forearm" \
#     --ergo_coeff 0.2 --resume 1 \
#     --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try2/Carry-timber-train-1/Humanoid_06-16-56-31/nn/Humanoid.pth  \


box w. handle v1
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_handle_construction.yaml \
    --num_envs 10240 \
    --headless \
    --wandb_project "TokenHSI-MMH-Train" \
    --wandb_mode "online" \
    --random_mode_equal_proportion False \
    --construction_experiment False \
    --random_density True \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_handle.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try3/Carry-handle-train-1/ \
    --wandb_name "Try3-Carry-handle-train-1" \
    --notes "indented box try, smooth wrist, median H" \
    --ergo_coeff 0.2 \
    --resume 1 \
    --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try3/Carry-handle-train-1/Humanoid_06-16-56-31/nn/Humanoid.pth

    # --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try3/Carry-handle-train-2/ \
    # --wandb_name Try3-Carry-handle-train-2 \
    # --notes resume-box-MMH-carry \
    # --ergo_coeff 0.2 \
    # --resume 1 \
    # --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try3/Carry-handle-train-2/Humanoid_20-13-28-16/nn/Humanoid.pth



# python -u ./tokenhsi/run.py --task HumanoidCarry --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction_exp4-noReward.yaml --num_envs 10240 --headless --wandb_project TokenHSI-MMH-Train --wandb_mode online --random_size False --random_density True --box_w 0.095 --box_l 1.8 --box_h 0.045 --random_mode_equal_proportion False --construction_experiment False --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
#     --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try2/Carry-timber-train-4/ --wandb_name Try2-Carry-timber-train-4 --notes "320hr when finish - scratch, 60s, timber reward, small imitation motion, longer forearm" --ergo_coeff 0.2 \
#     --resume 1 \
#     --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try2/Carry-timber-train-4/Humanoid_20-13-25-30/nn/Humanoid.pth \

# python ./tokenhsi/run.py --task HumanoidCarry --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction_exp3-smallMotion.yaml --num_envs 10240 --headless --wandb_project TokenHSI-MMH-Train --wandb_mode online --random_size False --random_density True --box_w 0.095 --box_l 1.8 --box_h 0.045 --random_mode_equal_proportion False --construction_experiment False --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
#     --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try2/Carry-timber-train-3/ --wandb_name Try2-Carry-timber-train-3 --notes "320hr when finish - scratch, 60s, box reward, longer forearm" --ergo_coeff 0.2 \
#     --resume 1 \
#     --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try2/Carry-timber-train-3/Humanoid_15-16-18-48/nn/Humanoid.pth \
