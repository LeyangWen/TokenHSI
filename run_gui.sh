# conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch

# MMH Box Carry Test
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
    --checkpoint output/custom_trained/MMH-Try1/Carry-box-train-5-wrist/Humanoid_28-03-16-52/nn/Humanoid.pth  \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.4 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 180 \
    --ergo_coeff 0.2 \
    --construction_experiment True \
# todo: v4 ref motion mpath
    --user_urdf "tokenhsi/data/assets/carry_box/indented_box.urdf"  \
    --skip_img \
    --headless \
    --record_headless

    # --ergo_sub_weight "20, 40, 40" \
    # --headless \
    # --record_headless
    --density 156.25 \
    


# First try
# --checkpoint output/custom_trained/MMH-Try1/Carry-box-train-1/Humanoid_07-01-42-37/nn/Humanoid.pth \

# silipper box
--checkpoint output/custom_trained/MMH-Try1/Carry-box-train-2/Humanoid_17-14-12-41/nn/Humanoid.pth \  # resume from sticky, use edge, but only after adjut
--checkpoint output/custom_trained/MMH-Try1/Carry-box-train-scratch-3/Humanoid_17-14-11-34/nn/Humanoid.pth \  #  scratch --> do not use edge


# add wrist dof
output/custom_trained/MMH-Try1/Carry-box-train-3-wrist/Humanoid_26-01-20-34/nn/Humanoid.pth \  # resume from scratch-3, add wrist dof
# added high example
output/custom_trained/MMH-Try1/Carry-box-train-5-wrist/Humanoid_28-03-16-52/nn/Humanoid.pth  \
# all above uses "mjcf/phys_humanoid_v4_box_foot_tall_slippery_og+wrist.xml"



# MMH Timber Test
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
    --checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-19/Humanoid_08-16-13-40/nn/Humanoid.pth \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.095 \
    --box_l 1.8 \
    --box_h 0.045 \
    --random_size False \
    --random_mode_equal_proportion False \
    --random_density True \
    --density 180 \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    
# wrong imitation motion
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-1/Humanoid_03-04-44-31/nn/Humanoid.pth  \
# corrected, but not working
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-2/Humanoid_04-03-51-55/nn/Humanoid.pth \
# bug fix
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-3/Humanoid_25-14-35-13/nn/Humanoid.pth \
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-4/Humanoid_26-19-00-28/nn/Humanoid.pth \
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-5/Humanoid_26-19-00-14/nn/Humanoid.pth \
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-6/Humanoid_28-18-12-21/nn/Humanoid.pth \
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-7/Humanoid_28-18-13-02/nn/Humanoid.pth \
# modified handheld reward to prompt reaching to target
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-8/Humanoid_30-00-11-04/nn/Humanoid.pth \
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-9/Humanoid_30-00-13-03/nn/Humanoid.pth \  # 0.25* disc reward
# throughts: hand now stuck on lower than box location --> rewason: timber_height_reward too much
# Can reach to target and hold properly, especially for low location. But can not lift --> reason: your reward is stuck after that, no motivation to keep lifting, need increasing new reward
# sometime do now bend, small issue, might be fixed by twicking weights
# some kicking behavivor
# TODO: NIOSH post processing--> V4 back angle calculation change
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-10/Humanoid_30-23-11-35/nn/Humanoid.pth \  # v3 reward, added pickup reward, no mono
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-11/Humanoid_30-23-15-13/nn/Humanoid.pth
# Inner hand orientation wrong, lets encofce them to point towards the axis, also no bad imitaion motion
# hand location right, but is not lifting at all, just holding, maybe increase lift reward weight
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-12/Humanoid_01-20-28-16/nn/Humanoid.pth
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-13/Humanoid_01-20-32-30/nn/Humanoid.pth
# 12 works the same, 13 worse,  wont bend. Maybe need longer training, Maybe need weight adjustment, 
# maybe the finger angle is not suitable for this task  --> horizontal finger xml
# maybe shouldn't hold too far apart, see example motion hand loc
# TODO: No kicking
# hand facing reward is not working correctly --> fixed
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-14/Humanoid_06-05-02-09/nn/Humanoid.pth
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-15/Humanoid_06-04-56-00/nn/Humanoid.pth
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-16/Humanoid_07-01-29-43/nn/Humanoid.pth # stop at lowest location, do not lift
# --> more imitation, include carry motion
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-17/Humanoid_08-06-30-32/nn/Humanoid.pth  # just not lifting at lowest, lift at higher loc
# --> hand face bad, but liftin gposture, no lift
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-18/Humanoid_08-06-30-49/nn/Humanoid.pth  # kind of lifts (scratch horz-hand; more omomo type motions)
--checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-19/Humanoid_08-16-13-40/nn/Humanoid.pth  # best so far--> lifts, but hand location is werid at end (resume horz-hand; more omomo type motions, 60s) 60 hrs
# thought: check if init use the lifting phrase --> seem so, just printed out to check, did not do in depth analysis
# thought: maybe we need carry example init to finish lifting  --> added as mono type motion only
# AMP check if wrist is used. 
# maybe it is about the weight of the reward, try a bunch. 
# wrist gap too big,--> made v4_hoz_hand forearm longer
# tought: try v4 on box and see if it works

# Issue: hand location too far forward after lift, pitch under arm, too high box final height
# thought: maybe skill des prob should be all lift at test
# reset time should definitely be long, maybe even > 60 s
# throught: maybe og reward it self would just work, train again with og reward, long time, and more imitation
# --> Learn from this, then start traiing box handle





# Handle Carry Test 
# 1. imitation motion w/ box annotation, done
# 2. new yaml file, done
# 3. carry box urdf with handle, done
# 4. collison check in amp, include box? have handel? --> seems no collision check, just rsi frame ranges

# Basic Carry Test
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-11/Humanoid_29-23-04-07/nn/Humanoid.pth \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.4 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 180 \
    --ergo_coeff 0.2 \
    --user_urdf "tokenhsi/data/assets/carry_box/indented_box.urdf"  \
    --skip_img \
    --headless \
    --record_headless

    # --ergo_sub_weight "20, 40, 40" \
    # --headless \
    # --record_headless
    --density 156.25 \
    



    # Try 4 famliy
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-ergoReward-scratch-train-8/Humanoid_28-16-04-43/nn/Humanoid.pth \ # Try4 -8, good lift, no lower - walk pass instead.
    # --checkpoint output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-9/Humanoid_28-16-04-15/nn/Humanoid.pth \ # Try4 -9, good lift and lower, sometime bad aim at lower and not stable,  drop box 
    # output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-9/Humanoid_08-15-58-26/nn/Humanoid.pth \ # Try4 -9, continues train, quite stable, werid sidestep when lifting, sometimes pick up multiple times
    # output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-9/Humanoid_23-03-39-52/nn/Humanoid.pth \ # Try4 -9, continues train, side step to carry ***
    # output/custom_trained/Try4/Carry-GoodMotion-resume-ergoReward-train-11/Humanoid_29-23-04-07/nn/Humanoid.pth \ # Try4 -11, continues train, reward bug fix --> CRC

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


# Homedepot cardboard box sizes
# # Small box
# --box_w 0.41 \
# --box_l 0.25 \
# --box_h 0.31 \

# # Medium box
# --box_w 0.51 \
# --box_l 0.38 \
# --box_h 0.41 \

# # Large box
# --box_w 0.66 \
# --box_l 0.38 \
# --box_h 0.41 \



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
    

# MMH Train local visualize
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_small.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction.yaml \
    --num_envs 24 \
    --wandb_project "TokenHSI-MMH-Train" \
    --wandb_mode "disabled" \
    --random_size True \
    --random_density True \
    --box_w 0.095 \
    --box_l 1.8 \
    --box_h 0.045 \
    --random_mode_equal_proportion False \
    --construction_experiment False \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
    --wandb_name "Carry-box-train-1" \
    --notes "box w. 1 good motion and 0.2 reward" \
    --ergo_coeff 0.2 \
    --resume 1 \
    --checkpoint output/custom_trained/MMH-Try1/Carry-timber-train-14/Humanoid_06-05-02-09/nn/Humanoid.pth 



    --box_w 0.095 \
    --box_l 1.8 \
    --box_h 0.045 \


--> thoguhts: w and h is swapped during init, but not in imitation init
- maybe fingre more friction, done
- maybe make hook angle bigger
-- maybe increase min lift height, done
--> if hand close is calcuated from box center, then it might be a problem for the long box



# MMH Train local visualize - handle
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_small.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_handle_construction.yaml \
    --num_envs 24 \
    --wandb_project "TokenHSI-MMH-Train" \
    --wandb_mode "disabled" \
    --random_mode_equal_proportion False \
    --construction_experiment False \
    --random_density True \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_handle.yaml \
    --wandb_name "Carry-box-train-1" \
    --notes "box w. 1 good motion and 0.2 reward" \
    --ergo_coeff 0.2 \
    --resume 1 \
    --checkpoint output/custom_trained/MMH-Try1/Carry-box-train-5-wrist/Humanoid_28-03-16-52/nn/Humanoid.pth













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
    --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \
    --checkpoint output/custom_trained/try9/TerrainObstacles-GoodMotion-Reward-scratch-1/Humanoid_23-03-57-15/nn/Humanoid.pth \
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
    --user_urdf "tokenhsi/data/assets/carry_box/indented_box.urdf"  \
    --skip_img \
    --headless \
    --record_headless \
#     --nums_terrains 0 \


# Terrain Carry Test, og env
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_slope.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \
    --checkpoint output/custom_trained/try9/TerrainObstacles-GoodMotion-Reward-scratch-1/Humanoid_23-03-57-15/nn/Humanoid.pth \
    --test \
    --num_envs 15 \
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
    
# Terrain Carry Test, slope env
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_slope_test.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \
    --checkpoint output/custom_trained/try9/Terrain-GoodMotion-Reward-scratch-4/Humanoid_29-23-25-41/nn/Humanoid.pth \
    --test \
    --num_envs 15 \
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
    --load_terrain False \
    --load_slopes True \
    --skip_img \
    --headless \
    --record_headless

# Try 9 family
# --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \
# --checkpoint output/custom_trained/try9/TerrainObstacles-GoodMotion-Reward-scratch-1/Humanoid_23-03-57-15/nn/Humanoid.pth \ # obstacles, motion + reward, 
# --checkpoint output/custom_trained/try9/Terrain-GoodMotion-Reward-scratch-2/Humanoid_23-03-56-14/nn/Humanoid.pth \ # motion + reward, good posture, stuck sometimes
# --checkpoint output/custom_trained/try9/Terrain-GoodMotion-scratch-3/Humanoid_23-03-55-05/nn/Humanoid.pth \ # motion only
# output/custom_trained/try9/Terrain-GoodMotion-Reward-scratch-4/Humanoid_29-23-25-41/nn/Humanoid.pth # motion+ reward continue, reward bug fix

# Try 8 family
# --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \

# --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_08-10-42-05/nn/Humanoid.pth \
# --checkpoint output/custom_trained/try8/TerrainObstacles-GoodMotion-Reward-resume-pretrained-5/Humanoid_23-03-58-31/nn/Humanoid.pth \ # obstacles, motion + reward, do not trip can MMH if min_h >0.05, bad lower ***

# --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_08-10-42-05/nn/Humanoid.pth \
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


# python lpanlib/others/video.py --imgs_dir "output/imgs/timber_hold_2" --video_name "vid"  --fps 10 --delete_imgs


# | Keyboard | Function |
# | ---- | --- |
# | F | focus on humanoid |
# | Right Click + WASD | change view port |
# | Shift + Right Click + WASD | change view port fast |
# | K | visualize lines |
# | L | record screenshot, press again to stop recording|


