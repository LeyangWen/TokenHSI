import sys
sys.path.append("./")
import os
import os.path as osp
import argparse
import numpy as np
import torch
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

def _arr_of(v):
    return v["arr"] if isinstance(v, dict) and "arr" in v else v

def _wrap_with_context(arr, dtype):
    a = np.asarray(arr, dtype=dtype)
    return {"arr": a, "context": {"dtype": str(a.dtype)}}

def load_ref(path):
    x = np.load(path, allow_pickle=True)
    data = x.item() if hasattr(x, "item") else x
    return {
        "rotation":                _arr_of(data.get("rotation")),
        "root_translation":        _arr_of(data.get("root_translation")),
        "global_velocity":         _arr_of(data.get("global_velocity")),
        "global_angular_velocity": _arr_of(data.get("global_angular_velocity")),
        "skeleton_tree":           data.get("skeleton_tree"),
        "is_local":                bool(data.get("is_local", True)),
        "fps":                     int(data.get("fps", 20)),
    }

def save_ref(path, motion: SkeletonMotion):
    # Use TokenHSI to_dict() to stay canonical (no global_* saved)
    d = {
        "rotation": motion.to_dict()["rotation"],
        "root_translation": motion.to_dict()["root_translation"],
        "global_velocity": motion.to_dict()["global_velocity"],
        "global_angular_velocity": motion.to_dict()["global_angular_velocity"],
        "skeleton_tree": motion.to_dict()["skeleton_tree"],
        "is_local": motion.to_dict()["is_local"],
        "fps": motion.to_dict()["fps"],
    }
    np.save(path, d)

def _wrap_with_context(arr, dtype):
    a = np.asarray(arr, dtype=dtype)
    return {"arr": a, "context": {"dtype": str(a.dtype)}}

def _norm_wrap(x, dtype):
    """
    Normalize TokenHSI tensor dicts:
      - If x is {'arr': ..., ...}, unwrap to the array.
      - Convert to desired dtype.
      - Return {'arr': ndarray, 'context': {'dtype': '<dtype>'}}.
    """
    if isinstance(x, dict) and "arr" in x:
        x = x["arr"]
    a = np.asarray(x, dtype=dtype)
    return {"arr": a, "context": {"dtype": str(a.dtype)}}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process Blender ref_motion to match TokenHSI skeleton offsets")
    parser.add_argument("--root", default="tokenhsi/data/dataset_carry/motions/MMH/box_pickUp/box01.51470934.20250919201305+__+clip_01/phys_humanoid_v3")
    args = parser.parse_args()
    
    old_ref_path = osp.join(args.root, "ref_motion.npy")
    new_blender_ref_path = osp.join(args.root, "blender", "ref_motion.npy")
    out_path = osp.join(args.root, "blender", "ref_motion.npy")
    
    old = load_ref(old_ref_path)
    newb = load_ref(new_blender_ref_path)
    
    # build fixed skeleton_tree (use NEW names/parents, OLD local_translation)
    sk_new = newb["skeleton_tree"]
    sk_old = old["skeleton_tree"]
    
    def _norm_wrap(x, dtype):
        if isinstance(x, dict) and "arr" in x:
            x = x["arr"]
        a = np.asarray(x, dtype=dtype)
        return {"arr": a, "context": {"dtype": str(a.dtype)}}
    
    sk_fixed = {
        "node_names": sk_new["node_names"],
        "parent_indices": _norm_wrap(sk_new["parent_indices"], np.int64),
        "local_translation": _norm_wrap(sk_old["local_translation"], np.float32),
    }
    skeleton_tree = SkeletonTree.from_dict(sk_fixed)
    
    # rebuild motion with Blender rotations & roots, TokenHSI velocities
    r = torch.from_numpy(newb["rotation"]).float()
    t = torch.from_numpy(newb["root_translation"]).float()
    is_local = bool(newb["is_local"])
    fps = int(newb["fps"])
    
    state = SkeletonState.from_rotation_and_root_translation(skeleton_tree=skeleton_tree, r=r, t=t, is_local=is_local)
    motion = SkeletonMotion.from_skeleton_state(state, fps=fps)
    
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    save_ref(out_path, motion)
    print(f"[OK] Wrote matched motion: {out_path}")