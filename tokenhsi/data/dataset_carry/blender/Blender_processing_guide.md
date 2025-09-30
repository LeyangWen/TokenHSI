Got it — here’s an updated **README.md** with your requested fixes applied:

---

# Blender → TokenHSI: Motion Build, Postprocess, and Render

This pipeline is meant to be run after you have executed:

python tokenhsi/data/dataset_carry/preprocess_smplest.py
python tokenhsi/data/dataset_carry/generate_motion.py


Those two scripts prepare the initial reference motions (ref_motion.npy) that Blender will take in and let you edit.

After that, follow the steps below to visualize, modify, and re-integrate motions using Blender.

---

## 0) Set your root paths (bash variables)

Pick your dataset **root** (the motion folder) and save it as a shell variable. Use **one** now; you can add more later.

```bash
# MMH box_pickUp example
ROOT="tokenhsi/data/dataset_carry/motions/MMH/box_pickUp/box01.51470934.20250919201305+__+clip_01/phys_humanoid_v3"
# MMH handle_pickUp example
ROOT="tokenhsi/data/dataset_carry/motions/MMH/handle_pickUp/handle01.51470934.20250919200241+__+clip_02/phys_humanoid_v3"
# MMH timber_pickUp example
ROOT="tokenhsi/data/dataset_carry/motions/MMH/timber_pickUp/timber01.51470934.20250919201355+__+clip_01/phys_humanoid_v3"

# Absolute variant (recommended for Blender scripts)
ABS_ROOT="/home/leyang/Documents/TokenHSI/${ROOT}"
echo $ABS_ROOT

# Humanoid XML used for render checks
HUMANOID_XML="tokenhsi/data/assets/mjcf/phys_humanoid_v3_box_foot_tall.xml"
```

---

Got it — you want the README to explicitly tell you how to **open Blender** and run each script by pointing to the **absolute path** of the `.py` files, so you don’t have to hunt for them. Here’s the updated version of those steps:

---

## 1) In Blender: build a simple humanoid

Open Blender, then run this script from the Python console inside Blender (or pass via `--python`):

```
# Inside Blender > Scripting tab
/home/leyang/Documents/TokenHSI/tokenhsi/data/dataset_carry/blender/blender_build_simple_humanoid.py
```

---


## 2) Pose editing & box setup (in Blender)

* Modify the motion as needed.
* Add the box object you’ll animate alongside the humanoid.
* **Keyframing hotkeys:**

  * `I` → choose **Location & Rotation** to insert a keyframe.
  * `X` → delete keyframes in the Dope Sheet / Timeline.
* Save your `.blend`.

---


## 3) In Blender: export the motion

Open Blender, then run this script (again from the Scripting tab or with `--python`):

```
/home/leyang/Documents/TokenHSI/tokenhsi/data/dataset_carry/blender/blender_export_isaacgym.py
```


That way you can literally **copy-paste the full path** into Blender’s Scripting editor without browsing.

Do you want me to also add a **blender CLI example** (like `blender --background --python ...`) side by side with these, so you can either run from inside Blender UI or from terminal?


This writes Blender outputs under:

```
$ROOT/blender/
  ├─ ref_motion.npy
  ├─ box_motion.npy
```

---

## 4) Postprocess in TokenHSI (conda env)

Activate your env:

```bash
conda activate tokenhsi
```

Run postprocessing:

```bash
python tokenhsi/data/dataset_carry/blender/postprocess_blender_output.py --root "$ROOT"
```

Expected outputs (under `$ROOT/blender/`):

```
ref_motion.npy              # overwritten with matched canonical skeleton
box_motion.npy              # box (T,7) = [x y z qx qy qz qw]
```

---

## 5) Render & check (TokenHSI)

```bash
python tokenhsi/data/dataset_carry/blender/motion_render_and_check.py \
  --root "$ROOT" \
  --humanoid-xml "$HUMANOID_XML" \
  --out-html "box_motion_render.html" \
  --box-size 0.095 1.8 0.045 
  # --box-size 0.34 0.34 0.38
```

This will produce:

```
$ROOT/blender/box_motion_render.html
```

Open it in a browser to visually check the motion.

---

## Notes & troubleshooting

* **FBX warning**:
  `Error: FBX library failed to load ... No module named 'fbx'`
  This is fine unless you explicitly use FBX import/export.

* **Velocity diffs**:
  Small numeric differences (`1e-6`–`1e-4`) are expected in `global_velocity` and `global_angular_velocity` because they are recomputed.

* **None fields**:
  The OLD file may have `global_rotation=None` and `global_translation=None`. These are OK to ignore.

---

## Script summary

* `blender_build_simple_humanoid.py` — create simple rig/scene (adjust `$ABS_ROOT` inside).
* `blender_export_isaacgym.py` — export motions to `$ROOT/blender/ref_motion.npy` and `box_motion.npy`.
* `postprocess_blender_output.py` — rebuild motion with canonical skeleton, overwrite `ref_motion.npy`.
* `motion_render_and check.py` — compare OLD vs NEW, render HTML, and print consistency checks.

---

Do you want me to also make you a **copy-paste workflow block** at the very end (0–5 combined), so you can run everything for a new motion with minimal editing?
