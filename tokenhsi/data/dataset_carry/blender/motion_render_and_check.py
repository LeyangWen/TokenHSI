import sys
sys.path.append("./")
import torch
import os
import os.path as osp
import argparse
import json
import numpy as np
import trimesh
from lpanlib.isaacgym_utils.vis.api import vis_hoi_use_scenepic_animation
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.others.colors import name_to_rgb


def arr_of(v):
    """Unwrap dicts like {'arr': ndarray} into ndarray; pass through ndarray."""
    if isinstance(v, dict) and "arr" in v:
        return v["arr"]
    return v


def load_ref_motion(path):
    x = np.load(path, allow_pickle=True)
    print(x.item().keys())
    # raise NotImplementedError("This script is deprecated. Please use the updated version in the repository.")
    data = x.item() if hasattr(x, "item") else x  # support dict-like or pure arrays
    # Normalize common fields
    ref = {
        "rotation":               arr_of(data.get("rotation")),
        "root_translation":       arr_of(data.get("root_translation")),
        "global_translation":     arr_of(data.get("global_translation")),
        "global_rotation":        arr_of(data.get("global_rotation")),
        "global_velocity":        arr_of(data.get("global_velocity")),
        "global_angular_velocity":arr_of(data.get("global_angular_velocity")),
        "fps":                    int(data.get("fps", 20)),
        "is_local":               bool(data.get("is_local", True)),
        "skeleton_tree":          data.get("skeleton_tree"),
    }
    # Basic presence checks
    if ref["rotation"] is None or ref["root_translation"] is None:
        raise RuntimeError(f"{path}: missing rotation/root_translation")
    return ref


def load_box_motion(path):
    # Expected shape: (T, 7) -> (x,y,z,qx,qy,qz,qw)
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise RuntimeError(f"{path}: expected shape (T,7), got {arr.shape}")
    return arr


def _unwrap_one(v):
    """Unwrap {'arr': ndarray} -> ndarray; else return v."""
    if isinstance(v, dict) and "arr" in v and isinstance(v["arr"], np.ndarray):
        return v["arr"]
    return v

def _summ(v):
    """Summary for header line."""
    v = _unwrap_one(v)
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape}, dtype={v.dtype}"
    if isinstance(v, dict):
        return f"dict[{len(v)}]"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    return type(v).__name__

def _as_ndarray(x):
    x = _unwrap_one(x)
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return None  # not an array-like


def _near_check(label, new_val, old_val, rtol=1e-4, atol=1e-4):
    """
    Visible PASS/FAIL comparison for arrays, scalars, and simple dicts.
    Prints metrics (shape, max|diff|, mean|diff|) when comparable.
    """
    nv = _unwrap_one(new_val)
    ov = _unwrap_one(old_val)

    # Arrays
    nva = _as_ndarray(nv)
    ova = _as_ndarray(ov)
    if nva is not None and ova is not None:
        if nva.shape != ova.shape:
            print(f"[CHECK][{label}] ❌ SHAPE MISMATCH  new{nva.shape} vs old{ova.shape}")
            return
        diff = nva - ova
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
        ok = np.allclose(nva, ova, rtol=rtol, atol=atol, equal_nan=True)
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"[CHECK][{label}] {status}  shape={nva.shape}  max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  (rtol={rtol}, atol={atol})")
        return

    # Numbers / bools / strings
    if isinstance(nv, (int, float, np.number, bool, str)) and isinstance(ov, (int, float, np.number, bool, str)):
        ok = (nv == ov) if isinstance(nv, (bool, str)) else (abs(float(nv) - float(ov)) <= max(atol, rtol*max(1.0, abs(float(ov)))))
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"[CHECK][{label}] {status}  new={nv}  old={ov}")
        return

    # Dicts (lightweight): compare keys + recurse a few known fields
    if isinstance(nv, dict) and isinstance(ov, dict):
        new_keys = set(nv.keys())
        old_keys = set(ov.keys())
        if new_keys != old_keys:
            print(f"[CHECK][{label}] ❌ DICT KEYS DIFFER  new{sorted(new_keys)} vs old{sorted(old_keys)}")
        else:
            print(f"[CHECK][{label}] ✅ DICT KEYS MATCH  {sorted(new_keys)}")
        # Special handling for skeleton_tree
        for subk in ("parent_indices", "local_translation"):
            if subk in nv and subk in ov:
                _near_check(f"{label}.{subk}", nv[subk], ov[subk])
        return

    # Fallback
    same = type(nv) == type(ov)
    status = "✅ PASS (type match)" if same else "❌ FAIL (type mismatch)"
    print(f"[CHECK][{label}] {status}  new_type={type(nv).__name__}  old_type={type(ov).__name__}")


def _print_value(label, val, side, indent=0):
    pad = "  " * indent
    val = _unwrap_one(val)

    print(f"{pad}{label} [{side}]  { _summ(val) }")

    if isinstance(val, np.ndarray):
        if val.size <= 10:
            print(f"{pad}  data =\n{pad}{val}")
        else:
            print(f"{pad}  data (first 2 rows):\n{pad}{val[:2]}")
            print(f"{pad}  ...")
            print(f"{pad}  data (60th rows):\n{pad}{val[59:60]}")
    elif isinstance(val, dict):
        for k in sorted(val.keys()):
            _print_value(f"{label}.{k}", val[k], side, indent+1)
    elif isinstance(val, (list, tuple)):
        print(f"{pad}  data = {val}")
    else:
        print(f"{pad}  data = {val}")

def dump_old_new(new_ref, old_ref, rtol=1e-5, atol=1e-6):
    """Print old vs new (shapes + sample data) and run near-match checks after each item."""
    def unwrap_top(d):
        if not isinstance(d, dict): 
            return d
        return {k: _unwrap_one(v) for k,v in d.items()}

    new_ref = unwrap_top(new_ref)
    old_ref = unwrap_top(old_ref)

    all_keys = sorted(set(new_ref.keys()) | set(old_ref.keys()))
    print("\n===== NEW vs OLD: FULL DUMP (shapes + sample data) =====\n")
    for k in all_keys:
        print(f"\n--- Key: {k} ---")
        if k in new_ref:
            _print_value(k, new_ref[k], side="NEW")
        else:
            print(f"[NEW] (missing key '{k}')")
        print("-"*40)
        if k in old_ref:
            _print_value(k, old_ref[k], side="OLD")
        else:
            print(f"[OLD] (missing key '{k}')")

        # After printing, run a visible near-match check when both sides exist
        if k in new_ref and k in old_ref:
            _near_check(k, new_ref[k], old_ref[k])

    print("\n===== END DUMP =====\n")
    # Keep your explicit parent_indices prints; add a check there too
    print(old_ref['skeleton_tree']['parent_indices'])
    print(new_ref['skeleton_tree']['parent_indices'])
    _near_check("skeleton_tree.parent_indices",
                new_ref['skeleton_tree']['parent_indices'],
                old_ref['skeleton_tree']['parent_indices'],
                rtol=0, atol=0)  # exact for indices


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ensure_globals_for_render(ref):
    """
    Ensure ref has numpy arrays for:
      - ref['global_translation'] (T,J,3)
      - ref['global_rotation']    (T,J,4, xyzw)
    If missing, rebuild via lpanlib from rotation/root_translation + skeleton_tree.
    Modifies and returns `ref`.
    """
    has_pos = ref.get("global_translation") is not None
    has_rot = ref.get("global_rotation") is not None
    if has_pos and has_rot:
        # unwrap dicts if needed
        ref["global_translation"] = _to_numpy(arr_of(ref["global_translation"]))
        ref["global_rotation"]    = _to_numpy(arr_of(ref["global_rotation"]))
        return ref

    skel_dict = ref.get("skeleton_tree")
    if skel_dict is None:
        raise RuntimeError("ref_motion missing 'skeleton_tree'; cannot reconstruct globals.")

    skeleton_tree = SkeletonTree.from_dict(skel_dict)
    r = _to_numpy(arr_of(ref["rotation"]))            # (T,J,4) local or global quats (xyzw)
    t = _to_numpy(arr_of(ref["root_translation"]))    # (T,3)
    is_local = bool(ref.get("is_local", True))
    fps = int(ref.get("fps", 20))

    # lpanlib expects torch
    r_t = torch.from_numpy(r).float()
    t_t = torch.from_numpy(t).float()

    state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=r_t,
        t=t_t,
        is_local=is_local,
    )
    motion = SkeletonMotion.from_skeleton_state(state, fps=fps)

    ref["global_translation"] = _to_numpy(motion.global_translation)   # (T,J,3)
    ref["global_rotation"]    = _to_numpy(motion.global_rotation)      # (T,J,4)

    return ref


def render_box_motion(humanoid_xml_path,
                      new_ref_motion,
                      box_motion,
                      out_html,
                      box_extents=(0.4, 0.4, 0.4),
                      T_sync=None):
    """
    Render HTML using vis_hoi_use_scenepic_animation.
    Requires: trimesh + lpanlib vis.
    Uses new_ref_motion['global_translation'] / ['global_rotation'] if available.
    Otherwise, falls back to root-only (and warns).
    """
    gpos = new_ref_motion.get("global_translation")
    grot = new_ref_motion.get("global_rotation")
    if gpos is None or grot is None:
        print("[WARN] new ref_motion has no global link poses; cannot render full humanoid. Skipping HTML.")
        return

    T = gpos.shape[0]
    if T_sync is None:
        T_sync = min(T, box_motion.shape[0])
    if T_sync < T:
        gpos = gpos[:T_sync]
        grot = grot[:T_sync]
    if T_sync < box_motion.shape[0]:
        box_motion = box_motion[:T_sync]

    # Build box mesh
    mesh = trimesh.creation.box(extents=list(box_extents))

    obj_meshes = [mesh]
    obj_global_pos = box_motion[:, np.newaxis, 0:3]  # (T,1,3)
    obj_global_rot = box_motion[:, np.newaxis, 3:7]  # (T,1,4)

    color_links = name_to_rgb['AliceBlue'] * 255
    color_box   = [name_to_rgb['LightYellow'] * 255]

    print(f"[Render] Writing HTML to: {out_html}")
    vis_hoi_use_scenepic_animation(
        asset_filename=humanoid_xml_path,
        rigidbody_global_pos=gpos,        # (T,J,3)
        rigidbody_global_rot=grot,        # (T,J,4)
        fps=int(new_ref_motion.get("fps", 20)),
        up_axis="z",
        color=color_links,
        output_path=out_html,
        obj_meshes=obj_meshes,
        obj_global_pos=obj_global_pos,
        obj_global_rot=obj_global_rot,
        obj_colors=color_box
    )
    print("[Render] Done.")


def main():
    ap = argparse.ArgumentParser(description="Post-check and render box motion")
    ap.add_argument("--root",        default="tokenhsi/data/dataset_carry/motions/MMH/box_pickUp/box01.51470934.20250919201305+__+clip_01/phys_humanoid_v3")
    ap.add_argument("--humanoid-xml",   default="tokenhsi/data/assets/mjcf/phys_humanoid_v3_box_foot_tall.xml")
    ap.add_argument("--box-size", type=float, nargs=3, default=(0.34, 0.34, 0.38), help="Box extents (X Y Z) in meters for rendering")
    ap.add_argument("--out-html", default="box_motion_render.html")
    args = ap.parse_args()
    args.file_old   = osp.join(args.root, "ref_motion.npy")
    args.file_new   = osp.join(args.root, "blender", "ref_motion.npy")
    args.box_motion = osp.join(args.root, "blender", "box_motion.npy")

    # Load
    print("[Load] OLD:", args.file_old)
    old_ref = load_ref_motion(args.file_old)
    print("[Load] NEW:", args.file_new)
    new_ref = load_ref_motion(args.file_new)
    new_ref = ensure_globals_for_render(new_ref)  # <-- add this line
    print("[Load] BOX:", args.box_motion)
    box = load_box_motion(args.box_motion)

    dump_old_new(new_ref, old_ref)  # prints everything; no checks/returns

    # Sanity check box length vs motion length
    if box.shape[0] != new_ref["rotation"].shape[0]:
        print(f"[WARN] box_motion T={box.shape[0]} vs new_ref T={new_ref['rotation'].shape[0]}; will clamp to min for render.")

    # Render HTML (unchanged)
    out_dir = osp.dirname(osp.abspath(args.file_new))
    os.makedirs(out_dir, exist_ok=True)
    out_html = osp.join(out_dir, args.out_html)
    render_box_motion(
        humanoid_xml_path=args.humanoid_xml,
        new_ref_motion=new_ref,
        box_motion=box,
        out_html=out_html,
        box_extents=tuple(args.box_size),
    )
    print("[INFO] Consistency checks assume UNMODIFIED Blender-exported motions. If your new files were retargeted/edited, differences are expected.")
    


if __name__ == "__main__":
    main()
