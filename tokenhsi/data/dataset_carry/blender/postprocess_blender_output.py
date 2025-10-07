#!/usr/bin/env python3
import sys
sys.path.append("./")

import os
import os.path as osp
import argparse
import numpy as np
import torch

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

# ---------- utils ----------
def _unwrap_arr(x):
    """Accept {'arr': ndarray, ...} or ndarray -> ndarray"""
    if isinstance(x, dict) and "arr" in x:
        return x["arr"]
    return x

def _wrap_arr(arr, dtype):
    a = np.asarray(arr, dtype=dtype)
    return {"arr": a, "context": {"dtype": str(a.dtype)}}

def load_any(path):
    """Load .npy (pickled dict) or .npz (keyed) into a flat dict."""
    if not osp.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            out = {}
            for k in z.files:
                v = z[k]
                if v.dtype == "O" and v.shape == ():  # nested dict packed as 0-d object
                    try:
                        out[k] = v.item()
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
            return out
    elif path.endswith(".npy"):
        return np.load(path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported file extension: {path}")

def load_ref(path):
    """Normalize a ref_motion dict to plain numpy arrays + metadata."""
    data = load_any(path)
    return {
        "rotation":                _unwrap_arr(data.get("rotation")),
        "root_translation":        _unwrap_arr(data.get("root_translation")),
        "global_velocity":         _unwrap_arr(data.get("global_velocity")),
        "global_angular_velocity": _unwrap_arr(data.get("global_angular_velocity")),
        "skeleton_tree":           data.get("skeleton_tree"),
        "is_local":                bool(data.get("is_local", True)),
        "fps":                     int(data.get("fps", 20)),
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Post-process Blender ref_motion to canonical TokenHSI format")
    ap.add_argument("--root", required=True,
                    help="Root folder of the motion (contains ref_motion.npy and blender/ref_motion.npy)")
    ap.add_argument("--old_name", default="ref_motion.npy", help="Old/canonical ref file name (under --root)")
    ap.add_argument("--new_rel", default="blender/ref_motion.npy", help="New Blender export relative path (under --root)")
    ap.add_argument("--out_rel", default="blender/ref_motion.npy", help="Output relative path (under --root)")
    args = ap.parse_args()

    old_ref_path = osp.join(args.root, args.old_name)
    new_ref_path = osp.join(args.root, args.new_rel)
    out_path     = osp.join(args.root, args.out_rel)

    print(f"[Load] OLD: {old_ref_path}")
    old = load_ref(old_ref_path)

    print(f"[Load] NEW: {new_ref_path}")
    newb = load_ref(new_ref_path)

    # Build fixed skeleton_tree: NEW node_names/parents + OLD local_translation (offsets)
    sk_new = newb["skeleton_tree"]
    sk_old = old["skeleton_tree"]

    if sk_new is None or sk_old is None:
        raise RuntimeError("Missing skeleton_tree in input files.")

    sk_fixed = {
        "node_names": sk_new["node_names"],
        "parent_indices": _wrap_arr(_unwrap_arr(sk_new["parent_indices"]), np.int64),
        "local_translation": _wrap_arr(_unwrap_arr(sk_old["local_translation"]), np.float32),
    }
    skeleton_tree = SkeletonTree.from_dict(sk_fixed)

    # Rebuild motion using Blender rotations + roots
    r = torch.from_numpy(newb["rotation"]).float()            # (T,J,4) local xyzw
    t = torch.from_numpy(newb["root_translation"]).float()    # (T,3)    world root
    is_local = bool(newb["is_local"])
    fps = int(newb["fps"])

    state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree, r=r, t=t, is_local=is_local
    )
    motion = SkeletonMotion.from_skeleton_state(state, fps=fps)

    # Save canonically via poselib (includes __name__ etc.)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    motion.to_file(out_path)
    print(f"[OK] Wrote matched motion: {out_path}")

if __name__ == "__main__":
    main()
