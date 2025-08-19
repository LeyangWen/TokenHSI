# conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch


# Basic Carry Test
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-3/Humanoid_07-01-14-58/nn/Humanoid.pth \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.4 \
    --box_l 0.4 \
    --box_h 0.4 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 156.25 \
    --ergo_coeff 0.2 \
    --skip_img \

    # --ergo_sub_weight "20, 40, 40" \
    # --headless \
    # --record_headless

    



    # Try 4 famliy
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-ergoReward-scratch-train-8/Humanoid_28-16-04-43/nn/Humanoid.pth \ # Try4 -8, good lift, no lower - walk pass instead.
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-9/Humanoid_28-16-04-15/nn/Humanoid.pth \ # Try4 -9, good lift and lower, sometime bad aim at lower and not stable,  drop box 
    # output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-9/Humanoid_08-15-58-26/nn/Humanoid.pth \ # Try4 -9, continues train, quite stable, werid sidestep when lifting, sometimes pick up multiple times ***
    # --checkpoint output/custom_trained/Try4/

    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-6/Humanoid_23-09-40-17/nn/Humanoid.pth \ # Try4 -6 # good lift, no lower - stop at target, I think the reward is higher then
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-ergoReward-scratch-train-7/Humanoid_23-09-37-42/nn/Humanoid.pth \ # Try4 -7, good lift, no lower - walk pass instead. also walking is slow
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-scratch-train-5/Humanoid_21-12-43-18/nn//Humanoid.pth  \ # Try4 -5,not working, not lifting, only hugging
    
    # above: bug fix on the ergo reward



    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-3/Humanoid_07-01-14-58/nn/Humanoid.pth \ Try4 -3 good for both lift and lower, some faliure on lowering, first train on good motion, then tune on good motion and ergo reward (actualy, ergo reward not properly used only motion) ***
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-train-2-resume/Humanoid_07-01-20-33/nn/Humanoid.pth \ # Try4 -2 good lift, but lower frequent fails, only trained on good motion
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ErgoReward-train-1/Humanoid_02-14-41-35/nn/Humanoid.pth \ Try 4 -1 resumed based on good one, with good motion and updated ergo, good motion but bad lift still
    # --checkpoint output/custom_trained/Try3/Carry-NewMotion-resume-ErgoReward-train-2/Humanoid_23-14-50-08/nn/Humanoid.pth \ Try3 -2 train more
    # --checkpoint output/custom_trained/Try3/Carry-NewMotion-resume-ErgoReward-train-2/Humanoid_18-00-28-19/nn/Humanoid.pth \  Try3 -2 0.2 ergo coeff
    # --checkpoint output/custom_trained/Try3/Carry-NewMotion-resume-ErgoReward-train-1/Humanoid_18-00-27-50/nn/Humanoid.pth \  Try3 -1 0.01 ergo coeff
    # --checkpoint output/custom_trained/Carry-train-2-1/Humanoid_07-04-13-24/nn/Humanoid.pth \  # new motion
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    # --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \

# --checkpoint output/custom_trained/Try1/Carry-train-8/Humanoid_27-23-50-50/nn/Humanoid.pth \
# --checkpoint output/custom_trained/Try1/Carry-train-6/Humanoid_27-23-51-50/nn/Humanoid.pth \
# --checkpoint output/custom_trained/Try1/Carry-train-8/Humanoid_27-23-50-50/nn/Humanoid.pth \
# --checkpoint output/custom_trained/Try1/Carry-train-10/Humanoid_29-03-42-24/nn/Humanoid.pth \ # best visual, have old motion  *** OG --> have faliure cases
# --checkpoint output/custom_trained/Try1/Carry-train-11/Humanoid_29-03-55-48/nn/Humanoid.pth \  # best reward, have old
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


# OG, no motion, no ergo reward
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint output/custom_trained/Try1/Carry-train-10/Humanoid_29-03-42-24/nn/Humanoid.pth \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.4 \
    --box_l 0.4 \
    --box_h 0.4 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 30.0 \
    --ergo_coeff 0.0 \
    --skip_img \



#########################################################################################################################
######################################################## Terrain ########################################################
#########################################################################################################################

# basic carry in terrain --> blind policy for terrain grid, trip over terrain
python -u ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --checkpoint /home/leyang/Documents/TokenHSI/output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-3/Humanoid_07-01-14-58/nn/Humanoid.pth  \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.4 \
    --box_l 0.4 \
    --box_h 0.4 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 200.0 \
    --ergo_coeff 0.2 \

# 


# Terrain Carry Test
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_08-10-42-05/nn/Humanoid.pth \
    --checkpoint output/custom_trained/try8/Terrain-GoodMotion-Reward-resume-pretrained-6/Humanoid_13-14-42-32/nn/Humanoid.pth \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "CarryTerrain_test" \
    --wandb_mode "disabled" \
    --notes "rand loc, test carry" \
    --box_w 0.40 \
    --random_size False \
    --random_density False \
    --density 100.0 \
    --random_mode_equal_proportion True \
    --construction_experiment True \
    --ergo_coeff 0.2 \
    --load_terrain True \

#     --nums_terrains 0 \


# Terrain Carry Test, og env
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_08-10-42-05/nn/Humanoid.pth \
    --checkpoint output/custom_trained/try8/Terrain-GoodMotion-resume-pretrained-7/Humanoid_09-06-20-48/nn/Humanoid.pth \
    --test \
    --num_envs 3 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "CarryTerrain_test" \
    --wandb_mode "disabled" \
    --notes "rand loc, test carry" \
    --box_w 0.40 \
    --random_size False \
    --random_density False \
    --density 100.0 \
    --random_mode_equal_proportion True \
    --construction_experiment False \
    --ergo_coeff 0.2 \
    --load_terrain False \



# Try 8 family
# --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_12-15-07-56/nn/Humanoid.pth \


# --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_08-10-42-05/nn/Humanoid.pth \
# --checkpoint output/custom_trained/try8/TerrainObstacles-GoodMotion-Reward-resume-pretrained-5/Humanoid_13-14-46-43/nn/Humanoid.pth \ # obstacles, motion + reward, do not trip can MMH if min_h >0.05, bad lower
# --checkpoint output/custom_trained/try8/Terrain-GoodMotion-Reward-resume-pretrained-6/Humanoid_13-14-42-32/nn/Humanoid.pth \ # motion + reward, Trips, can MMH, bad lower
# --checkpoint output/custom_trained/try8/Terrain-GoodMotion-resume-pretrained-7/Humanoid_09-06-20-48/nn/Humanoid.pth \ # motion only, unstable walk, can not MMH

# Terrain train 
python -m pdb ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --num_envs 1 \
    --box_w 0.4 \
    --random_size True \
    --random_density False \
    --random_mode_equal_proportion True \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "CarryTerrain_test" \
    --wandb_mode "disabled" \
    --ergo_coeff 0.0 \
    --load_terrain False \
    --headless


# OG Terrain
# python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
#     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
#     --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
#     --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
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


# python lpanlib/others/video.py --imgs_dir "output/imgs/Terrain_MMH_bad_lower" --video_name "vid" --delete_imgs --fps 10


