# conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch


# Basic Carry Test
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --checkpoint output/custom_trained/Carry-train-2-1/Humanoid_07-04-13-24/nn/Humanoid.pth \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.25 \
    --box_l 0.30 \
    --box_h 0.20 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 100.0 \



    # --checkpoint output/custom_trained/Carry-train-2-1/Humanoid_07-04-13-24/nn/Humanoid.pth \  # new motion
    
# --checkpoint output/custom_trained/Carry-train-8/Humanoid_27-23-50-50/nn/Humanoid.pth \
# --checkpoint output/custom_trained/Carry-train-6/Humanoid_27-23-51-50/nn/Humanoid.pth \
# --checkpoint output/custom_trained/Carry-train-8/Humanoid_27-23-50-50/nn/Humanoid.pth \
# --checkpoint output/custom_trained/Carry-train-10/Humanoid_29-03-42-24/nn/Humanoid.pth \ # best visual, have old
# --checkpoint output/custom_trained/Carry-train-11/Humanoid_29-03-55-48/nn/Humanoid.pth \  # best reward, have old
# sth in the code is making the box location in the floor
    # --checkpoint output/single_task/ckpt_carry.pth \

    # # Small box
    # --box_w 0.25 \
    # --box_l 0.30 \
    # --box_h 0.20 \

    # # Medium box
    # --box_w 0.35 \
    # --box_l 0.45 \
    # --box_h 0.30 \

    # # Large box
    # --box_w 0.45 \
    # --box_l 0.60 \
    # --box_h 0.40 \

    # # Extra large box
    # --box_w 0.60 \
    # --box_l 0.75 \
    # --box_h 0.50 \




# Terrain Carry Test
# python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
#     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
#     --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
#     --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
#     --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
#     --checkpoint output/tokenhsi/ckpt_stage2_terrainShape_carry.pth \
#     --test \
#     --num_envs 1 \
#     --wandb_project "TokenHSI-Test" \
#     --wandb_name "CarryTerrain_test" \
#     --wandb_mode "disabled" \
#     --notes "rand loc, test carry" \
#     --box_w 0.40 \
#     --random_size False \
#     --random_density False \
#     --random_mode_equal_proportion False \



# sh tokenhsi/scripts/single_task/traj_test.sh


# python lpanlib/others/video.py --imgs_dir "output/imgs/100_2" --video_name "density_100_40cm_cube.mp4" --delete_imgs 