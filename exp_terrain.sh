conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch

### MMH Box

# box_pth="output/custom_trained/MMH-Try1/Carry-box-train-6-wrist/Humanoid_13-22-39-59/nn/Humanoid.pth"
box_pth="output/custom_trained/MMH-Terrain-timber/Terrain-timber-scratch-1/Humanoid_17-00-37-27/nn/Humanoid.pth"

# box_s_10lbs
python ./tokenhsi/run.py --task HumanoidAdaptCarryGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_ground2terrain_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry_VEHS.yaml \
    --hrl_checkpoint output/custom_trained/try8/Try8-Stage1-GoodMotion-scratch-train-1/Humanoid_19-12-40-28/nn/Humanoid.pth \
    --checkpoint ${box_pth}  \
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

python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
    --checkpoint ${box_pth}  \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --box_w 0.34 \
    --box_l 0.34 \
    --box_h 0.36 \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --density 109 \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --skip_img \
    
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_construction.yaml \
    --num_envs 10240 \
    --headless \
    --wandb_project "TokenHSI-MMH-Train" \
    --wandb_mode "online" \
    --random_size True \
    --random_density True \
    --random_mode_equal_proportion False \
    --construction_experiment False \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_box.yaml \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/MMH-Try1/Carry-box-train-6-v5_armfix/ \
    --wandb_name "Carry-box-train-6-v5_armfix" \
    --notes "v5_armfix" \
    --ergo_coeff 0.2 \

### MMH Handle
handle_pth="output/custom_trained/MMH-Try3/Carry-handle-train-10/Humanoid_10-17-06-21/nn/Humanoid.pth"
# handle_s_10lbs
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_handle_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_handle.yaml \
    --checkpoint ${handle_pth} \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --random_size False \
    --random_mode_equal_proportion True \
    --random_density True \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --density 109 \
    --box_w 0.34 \
    --box_l 0.34 \
    --box_h 0.36 \
    --user_urdf "tokenhsi/data/assets/carry_box/indented_box_lab.urdf"  \
    --skip_img \



### MMH Timber
# timber_s
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_timber_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_timber.yaml \
    --checkpoint output/custom_trained/MMH-Try2/Carry-timber-train-1-v5_armfix/Humanoid_13-00-48-01/nn/Humanoid.pth   \
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
    --random_density True \
    --density 180 \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --skip_img \

### MMH bag


# bag_s
python -u ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/MMH/amp_humanoid_MMH_bag_construction.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_MMH_bag.yaml \
    --checkpoint output/custom_trained/MMH-Try4/Carry-bag-train-3/Humanoid_14-02-31-22/nn/Humanoid.pth     \
    --test \
    --num_envs 1 \
    --wandb_project "TokenHSI-Test" \
    --wandb_name "Carry_test_1" \
    --wandb_mode "disabled" \
    --random_size False \
    --random_density True \
    --ergo_coeff 0.2 \
    --construction_experiment True \
    --density 130 \
    --box_w 0.25 \
    --box_l 0.45 \
    --box_h 0.10 \
    --user_urdf "tokenhsi/data/assets/non_rigid_bag/concrete_bag.urdf"  \
    --skip_img \
#--density 1008


### Render
# python lpanlib/others/video.py --imgs_dir "output/imgs/handle_lift_works_0" --video_name "vid"  --fps 8 --delete_imgs


# | Keyboard | Function |
# | ---- | --- |
# | F | focus on humanoid |
# | Right Click + WASD | change view port |
# | Shift + Right Click + WASD | change view port fast |
# | K | visualize lines |
# | L | record screenshot, press again to stop recording|

