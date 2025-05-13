import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import argparse
from tqdm import tqdm
import csv

from tokenhsi.data.data_utils import process_VEHS7M_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "VEHS7M_start_end_time.csv"))
    parser.add_argument("--VEHS7M_DIR", type=str, default="/media/leyang/My Book/VEHS/VEHS-7M/Mesh/c3d")
    args = parser.parse_args()
    
    # load csv
    with open(args.dataset_cfg, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers = [h.lstrip("\ufeff") for h in headers]
        data = [dict(zip(headers, row)) for row in reader]
    print(headers)
    

    # CARRY_FILES = ["S01/Activity04_stageii",]
    
    
    # for i, file, in enumerate(CARRY_FILES):
    #     CARRY_FILES[i] = file.replace("/", "+__+")
        
    output_dir = os.path.join(os.path.dirname(__file__), "motions")
    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = {
        "carry": data,
    }
    omomo_add = []
    pickUp_add = []
    carryWith_add = []
    putDown_add = []
    
    for skill, data in candidates.items():
        output_dir_skill = os.path.join(output_dir, skill)
        os.makedirs(output_dir_skill, exist_ok=True)

        pbar = tqdm(data)
        for seq in pbar:
            smplx_file_name = f"{seq['subject']}/{seq['file'].split('.')[0]}_stageii.pkl"
            output_file_dir = smplx_file_name[:-4].replace("/", "+__+")+"+__+"+f"{seq['start']}-{seq['end']}_{seq['category']}"
            pbar.set_description(smplx_file_name)

   
            fname = os.path.join(args.VEHS7M_DIR, smplx_file_name)
            output_path = os.path.join(output_dir_skill, output_file_dir, "smpl_params.npy")

            os.makedirs(osp.dirname(output_path), exist_ok=True)
            
            process_VEHS7M_seq(fname, output_path, start_end=[seq["start"], seq["end"]])
            
            
            
            # make object info
            if seq['hand_level'] == '0' or seq['good_posture'] == '0':
                omomo_add.append(output_file_dir)
                continue
            if seq['category'] == 'pickUp':
                pickUp_add.append(output_file_dir)
            elif seq['category'] == 'carryWith':
                carryWith_add.append(output_file_dir)
            elif seq['category'] == 'putDown':
                putDown_add.append(output_file_dir)
                
    categories = {
        "pickUp": pickUp_add,
        "carryWith": carryWith_add,
        "putDown": putDown_add,
        "omomo": omomo_add,
    }

    # 2) for each non‐empty category, print the YAML block
    for skill_name, dirs in categories.items():
        if not dirs:
            continue
        print(f"{skill_name}:")
        for d in dirs:
            # build the two motion‑file paths
            ref = os.path.join("motions", "carry", d, "phys_humanoid_v3", "ref_motion.npy")
            obj = os.path.join("motions", "carry", d, "phys_humanoid_v3", "box_motion.npy")
            print(f"  - file: {ref}")
            if skill_name != "omomo":
              print(f"    obj_file: {obj}")
              print(f"    rsi_skipped_range: []")   # leave RSI blank
            print(f"    weight: 1.0")
        print()  # blank line between categories


"""
pickUp:
  - file: motions/carry/S01+__+Activity03_stageii+__+4991-5267_pickUp/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity03_stageii+__+4991-5267_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S01+__+Activity04_stageii+__+7348-7527_pickUp/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity04_stageii+__+7348-7527_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S01+__+Activity04_stageii+__+12342-12496_pickUp/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity04_stageii+__+12342-12496_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+733-1178_pickUp/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+733-1178_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity03_stageii+__+5881-6336_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity03_stageii+__+5881-6336_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity03_stageii+__+15146-15415_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity03_stageii+__+15146-15415_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity04_stageii+__+1073-1189_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity04_stageii+__+1073-1189_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity04_stageii+__+2607-2751_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity04_stageii+__+2607-2751_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity04_stageii+__+4815-5295_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity04_stageii+__+4815-5295_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity04_stageii+__+6704-6933_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity04_stageii+__+6704-6933_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity04_stageii+__+12981-13160_pickUp/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity04_stageii+__+12981-13160_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S04+__+Activity04_stageii+__+3588-3991_pickUp/ref_motion.npy
    obj_file: motions/carry/S04+__+Activity04_stageii+__+3588-3991_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S06+__+Activity04_stageii+__+8199-8319_pickUp/ref_motion.npy
    obj_file: motions/carry/S06+__+Activity04_stageii+__+8199-8319_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S10+__+Activity04_stageii+__+4168-4346_pickUp/ref_motion.npy
    obj_file: motions/carry/S10+__+Activity04_stageii+__+4168-4346_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0

carryWith:
  - file: motions/carry/S01+__+Activity05_stageii+__+10695-10851_carryWith/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity05_stageii+__+10695-10851_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+4851-5001_carryWith/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+4851-5001_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+5664-5853_carryWith/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+5664-5853_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+8899-9137_carryWith/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+8899-9137_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity04_stageii+__+12780-12926_carryWith/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity04_stageii+__+12780-12926_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity05_stageii+__+13505-13676_carryWith/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity05_stageii+__+13505-13676_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity03_stageii+__+8483-8675_carryWith/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity03_stageii+__+8483-8675_carryWith/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0

putDown:
  - file: motions/carry/S01+__+Activity04_stageii+__+555-1008_putDown/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity04_stageii+__+555-1008_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+1178-1400_putDown/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+1178-1400_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+5841-6069_putDown/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+5841-6069_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity03_stageii+__+9087-9266_putDown/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity03_stageii+__+9087-9266_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S02+__+Activity05_stageii+__+13621-13761_putDown/ref_motion.npy
    obj_file: motions/carry/S02+__+Activity05_stageii+__+13621-13761_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity03_stageii+__+6316-6726_putDown/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity03_stageii+__+6316-6726_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity03_stageii+__+15387-15762_putDown/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity03_stageii+__+15387-15762_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S03+__+Activity04_stageii+__+12676-12847_putDown/ref_motion.npy
    obj_file: motions/carry/S03+__+Activity04_stageii+__+12676-12847_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S07+__+Activity04_stageii+__+1591-1738_putDown/ref_motion.npy
    obj_file: motions/carry/S07+__+Activity04_stageii+__+1591-1738_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S10+__+Activity04_stageii+__+4442-4867_putDown/ref_motion.npy
    obj_file: motions/carry/S10+__+Activity04_stageii+__+4442-4867_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S10+__+Activity04_stageii+__+13985-14245_putDown/ref_motion.npy
    obj_file: motions/carry/S10+__+Activity04_stageii+__+13985-14245_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0

omomo:
  - file: motions/carry/S01+__+Activity03_stageii+__+1012-1410_pickUp/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity03_stageii+__+1012-1410_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S01+__+Activity04_stageii+__+1490-1751_putDown/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity04_stageii+__+1490-1751_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S01+__+Activity04_stageii+__+7912-8085_pickUp/ref_motion.npy
    obj_file: motions/carry/S01+__+Activity04_stageii+__+7912-8085_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S04+__+Activity04_stageii+__+449-1071_pickUp/ref_motion.npy
    obj_file: motions/carry/S04+__+Activity04_stageii+__+449-1071_pickUp/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
  - file: motions/carry/S04+__+Activity04_stageii+__+1071-1394_putDown/ref_motion.npy
    obj_file: motions/carry/S04+__+Activity04_stageii+__+1071-1394_putDown/box_motion.npy
    rsi_skipped_range: []
    weight: 1.0
"""