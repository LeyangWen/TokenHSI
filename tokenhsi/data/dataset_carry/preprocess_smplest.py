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
    args = parser.parse_args()

    # load yaml
    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # input/output dirs
    output_dir = os.path.join(os.path.dirname(__file__), "motions", "MMH")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = {
        "box_pickUp": cfg["motions"]["box_pickUp"],
        # "handle_pickUp": cfg["motions"]["handle_pickUp"],
        # "bag_pickUp": cfg["motions"]["bag_pickUp"],
        # "timber_pickUp": cfg["motions"]["timber_pickUp"],
    }

    for skill, data in candidates.items():
        output_dir_skill = os.path.join(output_dir, skill)
        os.makedirs(output_dir_skill, exist_ok=True)

        pbar = tqdm(data)
        for seq in pbar:

            pbar.set_description(seq)

            fname = seq  # "/home/leyang/Documents/SMPLest-X/demo/result_imitation_motions/Lift/good/clips/bag01.66920734.20250919201429/clip_01.pkl"
            # only keep after Lift, change / to +__+
            output_name = seq.split("clips/")[-1].replace("/", "+__+")
            output_path = os.path.join(output_dir_skill, output_name[:-4], "smpl_params.npy")

            os.makedirs(osp.dirname(output_path), exist_ok=True)
            
            process_smplest_seq(fname, output_path)

        print("Processed {} sequences!".format(len(data)))
