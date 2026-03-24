conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch

hrl_checkpoint="output/custom_trained/Stage1/Stage1-GoodMotion-scratch-train-1/Humanoid_14-22-34-39/nn/Humanoid.pth"
# hrl_checkpoint="output/custom_trained/Stage1/Stage1-GoodMotion-scratch-train-1/Humanoid_10-04-34-17/nn/Humanoid.pth"
### MMH Box
box_pth="output/custom_trained/MMH-Terrain-box/Terrain-box-scratch-1/Humanoid_21-16-40-13/nn/Humanoid.pth"

# box_s_10lbs
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
    --checkpoint ${box_pth}  \
    --hrl_checkpoint ${hrl_checkpoint} \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "CarryTerrain_test" \
    --wandb_mode "disabled" \
    --notes "rand loc, test carry" \
    --box_w 0.34 \
    --box_l 0.34 \
    --box_h 0.36 \
    --random_size False \
    --random_density False \
    --density 109 \
    --random_mode_equal_proportion False \
    --construction_experiment True \
    --ergo_coeff 0.2 \
    --load_terrain True \
    --skip_img \


### MMH Handle
handle_pth="output/custom_trained/MMH-Terrain-handle/Terrain-handle-scratch-1/Humanoid_21-16-40-13/nn/Humanoid.pth"
# handle_s_10lbs
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_handle.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_handle.yaml \
    --checkpoint ${handle_pth}  \
    --hrl_checkpoint ${hrl_checkpoint} \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --random_size False \
    --random_mode_equal_proportion False \
    --random_density False \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --load_terrain True \
    --density 109 \
    --box_w 0.34 \
    --box_l 0.34 \
    --box_h 0.36 \
    --user_urdf "tokenhsi/data/assets/carry_box/indented_box_lab.urdf"  \
    --skip_img \



### MMH Timber
# timber_s
timber_pth="output/custom_trained/MMH-Terrain-timber/Terrain-timber-scratch-1/Humanoid_21-16-40-13/nn/Humanoid.pth"
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_timber.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
    --checkpoint ${timber_pth}  \
    --hrl_checkpoint ${hrl_checkpoint} \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.095 \
    --box_l 0.915 \
    --box_h 0.045 \
    --random_size False \
    --random_mode_equal_proportion False \
    --random_density False \
    --density 180 \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --load_terrain True \
    --skip_img \

### MMH bag


# bag_s
bag_pth="output/custom_trained/MMH-Terrain-bag/Terrain-bag-scratch-1/Humanoid_21-16-40-13/nn/Humanoid.pth"
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction_bag.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_bag.yaml \
    --checkpoint ${bag_pth}  \
    --hrl_checkpoint ${hrl_checkpoint} \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --random_size False \
    --random_density False \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --density 130 \
    --box_w 0.25 \
    --box_l 0.45 \
    --box_h 0.10 \
    --user_urdf "tokenhsi/data/assets/non_rigid_bag/concrete_bag.urdf"  \
    --load_terrain False \
    --skip_img \
#--density 1008


### Render
# python lpanlib/others/video.py --imgs_dir "output/imgs/timber" --video_name "vid"  --fps 10 --delete_imgs


# | Keyboard | Function |
# | ---- | --- |
# | F | focus on humanoid |
# | Right Click + WASD | change view port |
# | Shift + Right Click + WASD | change view port fast |
# | K | visualize lines |
# | L | record screenshot, press again to stop recording|

