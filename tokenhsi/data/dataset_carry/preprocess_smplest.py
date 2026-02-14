import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import argparse
from tqdm import tqdm

from tokenhsi.data.data_utils import process_smplest_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../dataset_cfg_MMH.yaml"))
    parser.add_argument("--flip_hand", type=str, default='False', help="select from 'Left', 'Right', 'LeftRight', or False")
    args = parser.parse_args()
    """
    For timber, flip left hand, change output_path split
    
    """
    # load yaml
    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # input/output dirs
    output_dir = os.path.join(os.path.dirname(__file__), "motions", "MMH")

    os.makedirs(output_dir, exist_ok=True)

    # TODO:selected motions here
    candidates = {
        # "box_pickUp": cfg["motions"]["box_pickUp"],
        # "handle_pickUp": cfg["motions"]["handle_pickUp"],
        # "handle_carry": cfg["motions"]["handle_carry"],
        "bag_pickUp": cfg["motions"]["bag_pickUp"],
        "bag_carry": cfg["motions"]["bag_carry"],
        # "timber_pickUp": cfg["motions"]["timber_pickUp"],
        # "timber_carry": cfg["motions"]["timber_carry"],
        # "timber_putDown": cfg["motions"]["timber_putDown"],
    }
    target_fps = 20
    # candidates = {
    #     "dashboard_pickUp": cfg["motions"]["dashboard_pickUp"],
    #     "bad_pickUp": cfg["motions"]["bad_pickUp"],
    # }
    # target_fps = 30

    for skill, data in candidates.items():
        output_dir_skill = os.path.join(output_dir, skill)
        os.makedirs(output_dir_skill, exist_ok=True)

        pbar = tqdm(data)
        for seq in pbar:

            pbar.set_description(seq)

            fname = seq  # "/home/leyang/Documents/SMPLest-X/demo/result_imitation_motions/Lift/good/clips/bag01.66920734.20250919201429/clip_01.pkl"
            # only keep after Lift, change / to +__+
            # TODO: change split name according to file structure
            output_name = seq.split("clips/")[-1].replace("/", "+__+")
            # output_name = seq.split("timber")[-1].replace("/", "+__+")
            output_path = os.path.join(output_dir_skill, output_name[:-4], "smpl_params.npy")

            os.makedirs(osp.dirname(output_path), exist_ok=True)
            
            process_smplest_seq(fname, output_path, target_fps=target_fps, visualize=False, flip_hand=args.flip_hand)

        print("Processed {} sequences!".format(len(data)))
