# TokenHSI — WMSD / MMH Lifting Fork — Handover Guide

**Author of this fork:** Leyang Wen (DPM Lab, University of Michigan)
**Base repo:** [liangpan99/TokenHSI](https://github.com/liangpan99/TokenHSI) (CVPR 2025)
**This fork:** [LeyangWen/TokenHSI](https://github.com/LeyangWen/TokenHSI)

This document is a handover guide for the next person picking up this codebase. The upstream README (preserved in `README.md`) covers installation and the original TokenHSI demos; this file documents what was *added* and *modified* for the Work-related Musculoskeletal Disorder (WMSD) risk-reduction experiments, how to run them, and the traps to avoid.

---

## 1. What this fork does

The upstream TokenHSI codebase learns physics-based human-scene interaction skills (trajectory following, sitting, climbing, carrying) using AMP (Adversarial Motion Priors) in Isaac Gym. This fork extends the **carrying** skill into **four Manual Material Handling (MMH) lifting tasks** relevant to construction ergonomics, and adds a **WMSD ergonomics reward** to shape motions toward lower biomechanical risk.

The four MMH tasks are:

1. **Container / Box** — regular rigid box, lifted with palms on the sides
2. **Container with handles** — rigid box with indented side handles (URDF with handle geometry)
3. **Long lumber / timber** — long thin rigid object carried on one shoulder or in both arms
4. **Non-rigid bag** — multi-link soft bag (concrete bag, stone bag) simulating a deformable load

Each task has its own env config, motion dataset, URDF asset(s), and pre-trained checkpoint.



---

## 3. Key files to read first

To get productive quickly, read these files in order. They cover ~90% of what was modified in this fork.

### 3.1 Entry-point shell scripts (the "what to run" layer)

| File | Purpose |
|---|---|
| `exp.sh` | Flat-ground inference for all 4 MMH tasks × sizes × weights. Each block sets `--box_w/l/h`, `--density`, `--user_urdf`, `--checkpoint`. Start here to understand the CLI surface. |
| `exp_terrain.sh` | Same structure as `exp.sh` but for terrain-adaptation experiments. Uses the terrain-adapted checkpoints. |
| `run_gui.sh` | Local GUI run convenience wrapper. (NOTE: raw version, no need to care about.) |

### 3.2 Training scripts (Slurm on Great Lakes)

All of these are sbatch files targeted at UMich Great Lakes (`shdpm98` account, `spgpu` partition). You will need to change the `--account`, `--partition`, and output paths for your own cluster.

| File | Trains |
|---|---|
| `slurm_train_MMH.sh` | Box (container) — flat ground |
| `slurm_train_MMH_handle.sh` | Box with handles — flat ground |
| `slurm_train_MMH_timber.sh` | Long lumber — flat ground |
| `slurm_train_MMH_bag.sh` | Non-rigid bag — flat ground |
| `slurm_train_terrain.sh` | Box — terrain adaptation |
| `slurm_train_terrain_handle.sh` | Box with handles — terrain |
| `slurm_train_terrain_timber.sh` | Timber — terrain |
| `slurm_train_terrain_bag.sh` | Bag — terrain |
| `slurm_train_stage1.sh` | Upstream stage-1 foundational-skill training (unmodified from upstream workflow) NOTE: This code should be run before running the above 4 train_terrain.sh files; 4-5 days to run. |
| `slurm_train.sh`, `slurm_test.sh` | Generic wrappers |

Each slurm file does roughly: activate conda, set `LD_LIBRARY_PATH` for Isaac Gym, then call `./tokenhsi/run.py` with training args. The most important training args are `--num_envs 10240`, `--headless`, `--random_size True`, `--random_density True`, `--ergo_coeff 0.2`, `--motion_file ...`, `--checkpoint ...` (for resume), and `--output_path ...`.

### 3.3 Core reward / task logic

**The single most important file to read:**

`tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py`

This is the carrying task class. All of the WMSD / ergonomics reward design lives here. Specifically look at `_compute_reward` and the helper methods computing:

- Joint-angle based ergonomic penalties (back flexion, knee, wrist)
- Box/load position and orientation relative to pelvis
- Contact rewards for hands on grip points
- The `--ergo_coeff` weighting

Also modified in this file:
- `_build_box` / `_reset_box` — start location, size, density, random vs fixed
- `_reset_task` — target drop-off location (`self._tar_pos`)
- `_load_box_asset` — reads custom URDF from `--user_urdf` CLI arg (for handle box, bag, timber)
- `render` — screenshot saving for headless mode

Companion files:

| File | Role |
|---|---|
| `tokenhsi/env/tasks/adapt_interaction_skills/humanoid_adapt_carry_ground2terrain.py` | Terrain-adapted carry task. Mirrors the flat-ground reward modifications and adds terrain height-map observations. |
| `tokenhsi/env/tasks/humanoid.py` | Base humanoid class (camera, root state, headless render plumbing). Minor edits. |
| `tokenhsi/env/tasks/base_task.py` | Isaac Gym `render` / `_physics_step`. Headless screenshot saving hooked in here. |
| `tokenhsi/run.py` | Entry point. New CLI args are parsed here: `--box_w/l/h`, `--random_size`, `--random_density`, `--density`, `--ergo_coeff`, `--construction_experiment`, `--user_urdf`, `--random_mode_equal_proportion`, `--skip_img`, `--wandb_*`. |

### 3.4 Config files (YAML)

Env configs for the four MMH tasks:

```
tokenhsi/data/cfg/MMH/
├── amp_humanoid_MMH_construction.yaml          # box
├── amp_humanoid_MMH_handle_construction.yaml   # box with handles
├── amp_humanoid_MMH_timber_construction.yaml   # long lumber
└── amp_humanoid_MMH_bag_construction.yaml      # non-rigid bag
```

Train config (shared across tasks):
- `tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml`

Motion datasets (reference motions + object trajectories):
```
tokenhsi/data/dataset_carry/
├── dataset_MMH_box.yaml
├── dataset_MMH_handle.yaml
├── dataset_MMH_timber.yaml
└── dataset_MMH_bag.yaml
```

Each dataset YAML lists motion clip files with start/end frame indices and weights. Motion clips were sourced from the VEHS-6.7M dataset (see Section 6 on the imitation pipeline).

### 3.5 Custom URDF assets

```
tokenhsi/data/assets/carry_box/
├── indented_box_lab.yaml       # small box with handles
├── indented_box_lab_l.urdf     # large box with handles
tokenhsi/data/assets/non_rigid_bag/
├── concrete_bag.urdf           # small bag, multi-link
└── stone_bag_l.urdf            # large bag, multi-link
```

The humanoid skeleton is `tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml`. Joint hard limits (especially y-axis wrist / arm limits) were tightened here to stop the arms from hyperextending during lifting. Don't naively revert these — training with looser limits collapses into unnatural poses.

---

## 4. CLI arguments added in this fork

These are the flags you'll see in `exp.sh` / `exp_terrain.sh` / the slurm files. They extend upstream's `tokenhsi/run.py`.

| Flag | Purpose |
|---|---|
| `--box_w`, `--box_l`, `--box_h` | Fixed object dimensions in meters (width, length, height). Only used when `--random_size False`. |
| `--random_size` | If True, sample size each episode from the range defined in the env YAML. |
| `--random_mode_equal_proportion` | If True and random_size is True, scales all 3 dims together (keeps aspect ratio). |
| `--random_density` | If True, sample density each episode. If False, use `--density`. |
| `--density` | Fixed density in kg/m³. Combined with box dims to determine mass. Keep in the range the model was trained on; very heavy boxes break training/inference. |
| `--ergo_coeff` | Weight on the ergonomics reward term (default 0.2). Higher = more aggressive posture shaping, but too high collapses the motion. |
| `--construction_experiment` | Flag that flips in the construction-scene experiment configuration (start/end positions, platform heights). Set `True` for `exp.sh` runs, `False` for free training. |
| `--user_urdf` | Path to a custom URDF to load instead of the generated box primitive. Used for handle boxes, bags, and timber. |
| `--skip_img` | Skip per-frame screenshot saving. Keep this on unless you actually need the rendered frames — saving is slow. |
| `--wandb_project`, `--wandb_name`, `--wandb_mode` | Weights & Biases logging. `--wandb_mode "disabled"` turns it off. |
| `--notes` | Free-form string attached to the wandb run. |
| `--output_path` | Where to save checkpoints during training. |
| `--resume 1 --checkpoint <path>` | Resume training from a checkpoint. Used heavily — most of the good checkpoints were iteratively resumed across multiple slurm submissions. |

---

## 5. Training on Great Lakes (UMich Slurm)

The training scripts assume UMich Great Lakes. Notable choices:

- **Partition:** `spgpu` works. `gpu_mig40` triggers a Vulkan error. Plain `gpu` often segfaults.
- **Headless:** `--headless` is required on Slurm (no display). Inside `base_task.py`, `self.graphics_device_id` must stay `-1` when running headless or `create_sim` will segfault. See debug notes in upstream README.
- **LD_LIBRARY_PATH:** must point to both `$CONDA_PREFIX/lib` (for `libpython3.8.so.1.0`) and the Isaac Gym bindings directory (for `libmem_filesys.so`). Both exports are in every slurm script — don't remove them.
- **Python version:** conda env must be Python 3.8. Do **not** `module load python3.10-anaconda` or `module load pytorch` before activating the env — it will shadow the right Python.
- **Num envs:** 10240 for flat ground. Terrain runs use fewer (~2048) because the height-map observation is memory-heavy.
- **Training time:** ~60–80 hours per task to get a good policy. Plan on multiple resume-from-checkpoint submissions.

### Monitoring training

Weights & Biases is the primary monitoring tool (`--wandb_project "TokenHSI-MMH-Train"`). TensorBoard also works:

### Checkpoint layout

During training, checkpoints are saved to `--output_path` with a timestamped Humanoid directory, e.g.:

```
/scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/
└── MMH-Try1/
    └── Carry-box-train-6-v5_armfix/
        └── Humanoid_13-01-05-28/
            └── nn/
                └── Humanoid.pth
```

The final checkpoints referenced in `exp.sh` live under `output/custom_trained/MMH-TryN/...`. Those are copies pulled down from `/scratch` and committed (or symlinked) locally. If a checkpoint path is broken, check `/scratch` first.

---

## 6. Motion data pipeline (imitation / mocap → TokenHSI)

NOTE: This section will be used for `imitation learning`.

Reference motions come from the **VEHS-6.7M dataset** as well as smartphone videos (captured in the DPM lab). The full pipeline for processing a raw mocap clip into something TokenHSI can use as an AMP reference is documented here:

**→ [`tokenhsi/data/dataset_carry/blender/Blender_processing_guide.md`](https://github.com/LeyangWen/TokenHSI/blob/main/tokenhsi/data/dataset_carry/blender/Blender_processing_guide.md)**

Read that guide end-to-end before you try to add new reference motions. Short version of what it covers:

1. In the VEHS-6.7M dataset, annotate timestamps for start/end of clean carry motions in a CSV.
2. Export SMPL-X parameters for those segments (custom script in `vicon-read`).
3. `tokenhsi/data/dataset_carry/preprocess_amass.py` — converts frame rate and writes `smpl_params.npy`.
4. `tokenhsi/data/dataset_carry/generate_motion.py` — cuts SMPL motion into clips using CSV indices, writes `ref_motion.npy` and renders a preview.
5. `tokenhsi/data/dataset_carry/generate_object.py` — uses the same indices to generate the corresponding box trajectory (`box_motion.npy`) and render.
6. Add the new clip to `tokenhsi/data/dataset_carry/dataset_MMH_<task>.yaml` with a weight for the reference-state-initialization sampler (`sample_time_rsi`).

**Critical lesson learned:** reference-motion quality matters *enormously*. A handful of clean, representative carry clips beats dozens of noisy ones. Aggressively delete bad clips from the YAML. The reward function does *not* rescue bad reference data.

---

## 7. Reward design — where to look and what to change

All reward terms for the carry/MMH task live in:

**`tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py`**

The reward is a weighted sum of:

- **Imitation (AMP)** — standard AMP discriminator reward from the reference motion
- **Task reward** — distance from box to target platform, box-lifted-off-ground, box-on-target
- **Ergonomics reward** — this is the WMSD piece; it penalizes high-risk postures based on joint angles (back flexion, knee flexion, shoulder abduction, wrist deviation). Weighted by `--ergo_coeff`.
- **Regularization** — energy, joint velocity, contact with wrong body parts.

The humanoid skeleton index mapping (useful when reading reward code):

```
 0: pelvis              8: left_hand
 1: torso               9: right_thigh
 2: head               10: right_shin
 3: right_upper_arm    11: right_foot
 4: right_lower_arm    12: left_thigh
 5: right_hand         13: left_shin
 6: left_upper_arm     14: left_foot
 7: left_lower_arm
```

### Known reward bugs that were fixed (check you haven't regressed them)

From the author's notes:

- **20250722:** The ergo reward was not actually being applied (coefficient was zero-pathed somewhere). Fixed. If ergo-coeff > 0 but the motions look identical to coeff = 0, check this first.
- **20250722:** A separate reward term was preventing the humanoid from lowering the box onto the target. Fixed.

### Terrain-specific notes

`tokenhsi/env/tasks/adapt_interaction_skills/humanoid_adapt_carry_ground2terrain.py` mirrors the flat reward but adds terrain height-map observations. The terrain adaptation approach follows AdaptNet (Pei Xu) — a small adapter on top of the pretrained flat-ground MLP policy, rather than retraining from scratch. Reasons it might not work out of the box:

- `loadTerrain` bool in the YAML sometimes fails to override — may need to set it directly on `cfg["env"]["terrain"]["loadTerrain"]` in code.
- `mapLength` / `mapWidth` must be reduced for local-GPU debugging (default values OOM).
- `random_density` must be **False** during terrain adaptation because the pretrained policy's observation dimension was fixed without the density channel.
- Humanoid sometimes "flies off" at episode start — usually a start-location initialization bug, not a policy bug.

---

## 8. File-structure cheat sheet

```
TokenHSI/
├── README.md                              # upstream README + raw debug log (keep for install steps)
├── HANDOVER.md                            # ← this file
│
├── exp.sh                                 # inference on flat ground for all 4 MMH tasks
├── exp_terrain.sh                         # inference on terrain for all 4 MMH tasks
├── run_gui.sh                             # local GUI runner
│
├── slurm_train_MMH.sh                     # flat-ground training: box
├── slurm_train_MMH_handle.sh              # flat-ground training: box+handle
├── slurm_train_MMH_timber.sh              # flat-ground training: timber
├── slurm_train_MMH_bag.sh                 # flat-ground training: bag
├── slurm_train_terrain.sh                 # terrain training: box
├── slurm_train_terrain_handle.sh          # terrain training: box+handle
├── slurm_train_terrain_timber.sh          # terrain training: timber
├── slurm_train_terrain_bag.sh             # terrain training: bag
├── slurm_train_stage1.sh                  # upstream stage-1 foundational skills
├── slurm_train.sh, slurm_test.sh          # generic wrappers
│
├── tokenhsi/
│   ├── run.py                             # entry point; all CLI args live here
│   ├── env/tasks/
│   │   ├── basic_interaction_skills/
│   │   │   └── humanoid_carry.py          # ★ MMH reward + box/bag/timber logic
│   │   ├── adapt_interaction_skills/
│   │   │   └── humanoid_adapt_carry_ground2terrain.py   # ★ terrain version
│   │   ├── humanoid.py                    # base humanoid (render, camera)
│   │   └── base_task.py                   # Isaac Gym base (render, physics step)
│   ├── data/
│   │   ├── cfg/
│   │   │   ├── MMH/                       # ★ env configs for 4 MMH tasks
│   │   │   └── train/rlg/                 # train configs
│   │   ├── dataset_carry/
│   │   │   ├── dataset_MMH_box.yaml       # ★ reference motion lists
│   │   │   ├── dataset_MMH_handle.yaml
│   │   │   ├── dataset_MMH_timber.yaml
│   │   │   ├── dataset_MMH_bag.yaml
│   │   │   ├── preprocess_amass.py
│   │   │   ├── generate_motion.py
│   │   │   ├── generate_object.py
│   │   │   └── blender/
│   │   │       └── Blender_processing_guide.md   # ★ mocap pipeline guide
│   │   └── assets/
│   │       ├── mjcf/phys_humanoid_v3.xml  # humanoid skeleton (joint limits tweaked)
│   │       ├── carry_box/                 # handle-box URDFs
│   │       └── non_rigid_bag/             # multi-link bag URDFs
│   ├── learning/                          # upstream RL code (amp_agent, transformer, etc.)
│   └── scripts/                           # upstream run scripts (kept for reference)
│
├── output/                                # local checkpoints, screenshots, imgs/
├── lpanlib/                               # upstream utilities (video.py, etc.)
├── body_models/smpl/                      # SMPL body models (you download these)
└── assets/                                # demo gifs
```

★ = files modified or added in this fork.

---

## 9. Gotchas, bugs, and things to not waste a week on

A condensed list of issues that were solved the hard way:

1. **Isaac Gym + Python 3.10** — doesn't work. Must use Python 3.8.
2. **`libpython3.8.so.1.0: cannot open shared object file`** — fix: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`.
3. **`libmem_filesys.so: cannot open shared object file`** — fix: `export LD_LIBRARY_PATH="<isaacgym>/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"`.
4. **Segfault on Slurm** — use `--headless` and keep `graphics_device_id = -1` for `create_sim`. Partition `spgpu`, not `gpu_mig40` (Vulkan) or plain `gpu`.
5. **`SetCameraLocation: could not find camera with handle -1`** — caused by `graphics_device_id = -1`. If you need cameras on Slurm, you're in for a fight. The workaround in this repo is to export 3D pose + object positions and render locally in matplotlib / Blender.
6. **Headless runs produce no output** — `--record` argument doesn't actually do anything in upstream. Screenshot saving was hooked into `render()` manually in `base_task.py` / `humanoid.py` / `humanoid_carry.py`. See `output/imgs/`.
7. **Training collapses / person flies off / fails to lift** — usually one of:
   - Box too big relative to the range in the training YAML
   - Density too high (very heavy box)
   - Random density range too wide
   - Reference motions for this task are noisy — clean the dataset YAML
   - `ergo_coeff` too high (> ~0.3 starts collapsing natural motion)
8. **Terrain inference from a flat-ground checkpoint** — works directly (the author confirmed this is the intended path), but humanoid will trip. For real performance, fine-tune with AdaptNet-style adaptation (see Section 7).
9. **Sudden z-axis spin in exported motion clips** — caused by root-rotation smoothing in the motion preprocessing. Visualize in `vis.py` to check — in-sim motion is usually fine; the spin is a visualization artifact.
10. **PyTorch3D** — only needed for the long-horizon demo (`stage2_longterm`), not for the MMH tasks. Skip it unless you're touching that demo.

---
## Terrain --> scanning

- Use Polycam & iphone to scan
- Use Blender to process into height map [script here](https://github.com/LeyangWen/vicon-read/blob/master/conversion_scripts/IsaacGym/blender_to_heightmap_script.py)
- Put path to height map in config files --> yaml files