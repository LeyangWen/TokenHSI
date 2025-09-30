# Export Isaac Gym 15-link motion + box pose from MJCF-built empties
# with fixed FPS=20, output dir //blender/, mkdirs, and in-Blender logging.
#
# Differences vs your original:
# - Compute skeleton_tree.local_translation from first-frame global positions (child - parent), root uses absolute.
# - Compute per-joint global_velocity (m/s) from global positions (Blender world pose).
# - Compute per-joint global_angular_velocity (rad/s) from quaternion deltas (Blender world rot).
# - Keep global_translation/global_rotation directly from Blender.
# - Keep box_poses directly from Blender.
#
# Conventions:
# - Quaternions are stored in xyzw order in arrays (to match your example usage).
# - Angular velocity uses axis*angle/dt from the relative quaternion between consecutive frames.

import bpy
import os
import numpy as np
from mathutils import Quaternion

# ----------------- Config (your requests) -----------------
PELVIS_EMPTY_NAME = "Body::pelvis"
BOX_OBJECT_NAME   = "Cube"
OUTPUT_DIR        = "//blender/"      # always inside a 'blender' subfolder next to the .blend
FPS               = 20                # force 20 fps, do not read from Blender

# Isaac-15 order
NODE_NAMES = [
    'pelvis', 'torso', 'head',
    'right_upper_arm','right_lower_arm','right_hand',
    'left_upper_arm','left_lower_arm','left_hand',
    'right_thigh','right_shin','right_foot',
    'left_thigh','left_shin','left_foot',
]

# Parent array must align with NODE_NAMES index order
PARENTS = np.array(
    [-1,  0,  1,
      1,  3,  4,
      1,  6,  7,
      0,  9, 10,
      0, 12, 13], dtype=np.int32
)

# Optional passthroughs to keep interface compatible
HUMANOID_XML_PATH = None
DO_SCENEPIC = False

# ----------------- In-Blender logger -----------------
def _get_text_log(name="ExporterLog"):
    txt = bpy.data.texts.get(name)
    if txt is None:
        txt = bpy.data.texts.new(name)
    return txt

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    _get_text_log().write(msg + "\n")

# --------------- Helpers ------------------
def world_matrix(obj):
    return obj.matrix_world

def local_quat_from_world(child_obj, parent_obj=None):
    """
    Return local rotation of child relative to parent as a Blender Quaternion (w,x,y,z).
    If parent is None, this returns world rotation.
    """
    Mw = world_matrix(child_obj)
    Ml = Mw if parent_obj is None else parent_obj.matrix_world.inverted() @ Mw
    return Ml.to_quaternion()  # Blender stores as (w, x, y, z)

def obj_world_pose(obj):
    """
    Return world translation and quaternion in xyzw array order for saving.
    """
    Mw = obj.matrix_world
    t = Mw.to_translation()
    q = Mw.to_quaternion()  # Blender Quaternion (w,x,y,z)
    # convert to xyzw ndarray
    return (
        np.array([t.x, t.y, t.z], dtype=np.float32),
        np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
    )

def _wxyz_to_xyzw(q):
    # Blender Quaternion gives (w,x,y,z)
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

def _xyzw_to_wxyz(q_xyzw):
    # array xyzw -> tuple wxyz
    return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))

import numpy as np

def _quat_delta_angular_velocity(q_prev_xyzw, q_curr_xyzw, dt):
    """
    TokenHSI-style angular velocity (rad/s) from two xyzw quaternions.

    Matches lpanlib's SkeletonMotion._compute_angular_velocity math:
      dq = q_curr * conj(q_prev)   (both normalized)
      angle, axis = quat_angle_axis(dq)
      omega = axis * (angle / dt)

    Notes:
    - Inputs: xyzw, unit (or near-unit) quaternions.
    - No hemisphere flip; product is normalized (like quat_mul_norm).
    - Returns np.ndarray shape (3,) in world/global frame if q_* are global.
    """
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float32)

    # --- helpers (xyzw) ---
    def _normalize(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        return (q / n) if n > 0.0 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def _conj_xyzw(q):
        x, y, z, w = q
        return np.array([-x, -y, -z,  w], dtype=np.float64)

    def _mul_xyzw(a, b):
        # Hamilton product for xyzw
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        x = aw*bx + ax*bw + ay*bz - az*by
        y = aw*by - ax*bz + ay*bw + az*bx
        z = aw*bz + ax*by - ay*bx + az*bw
        w = aw*bw - ax*bx - ay*by - az*bz
        return np.array([x, y, z, w], dtype=np.float64)

    def _angle_axis_from_xyzw(q):
        # q assumed normalized. q = [x,y,z,w] with w in [-1,1]
        x, y, z, w = q
        w = np.clip(w, -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(max(1.0 - w*w, 0.0))  # = sin(theta/2)
        if s > 0.0:
            axis = np.array([x/s, y/s, z/s], dtype=np.float64)
        else:
            # angle ~ 0: axis can be anything; keep vector part as-is
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return angle, axis

    # Normalize inputs (TokenHSI uses quat_mul_norm which normalizes the product;
    # normalizing inputs as well keeps it consistent and stable)
    q_prev = _normalize(q_prev_xyzw)
    q_curr = _normalize(q_curr_xyzw)

    # Relative rotation dq = q_curr * conj(q_prev), then normalize (quat_mul_norm)
    dq = _mul_xyzw(q_curr, _conj_xyzw(q_prev))
    dq = _normalize(dq)

    angle, axis = _angle_axis_from_xyzw(dq)
    omega = axis * (angle / dt)
    return omega.astype(np.float32)



# -------- Exporter using empties ----------
class EmptiesIsaacExporter:
    def __init__(self,
                 node_names,
                 parents,
                 pelvis_empty_name="Body::pelvis",
                 box_object_name="Cube",
                 fps=20,
                 output_dir="//blender/",
                 humanoid_xml_path=None,
                 do_scenepic=False):

        self.node_names = list(node_names)
        self.parents = np.array(parents, dtype=np.int32)
        self.fps = int(fps)
        self.output_dir = bpy.path.abspath(output_dir)
        self.humanoid_xml_path = humanoid_xml_path
        self.do_scenepic = do_scenepic

        # mkdir -p
        os.makedirs(self.output_dir, exist_ok=True)

        # look up pelvis & box
        self.pelvis = bpy.data.objects.get(pelvis_empty_name, None)
        if self.pelvis is None:
            raise RuntimeError(f"Pelvis empty '{pelvis_empty_name}' not found.")
        self.box_obj = bpy.data.objects.get(box_object_name, None)
        if self.box_obj is None:
            raise RuntimeError(f"Box object '{box_object_name}' not found.")

        # collect Body:: empties
        self.body_empties = {}
        for obj in bpy.data.objects:
            if obj.type == 'EMPTY' and obj.name.startswith("Body::"):
                key = obj.name.split("Body::", 1)[1]
                self.body_empties[key] = obj

        missing = [n for n in self.node_names if n not in self.body_empties]
        if missing:
            raise RuntimeError(f"Missing Body:: empties for nodes: {missing}")

        self.node_obj = {n: self.body_empties[n] for n in self.node_names}
        self.parent_obj = {}
        for i, n in enumerate(self.node_names):
            pidx = self.parents[i]
            self.parent_obj[n] = None if pidx < 0 else self.node_obj[self.node_names[pidx]]

        log("[Init] Output dir:", self.output_dir)
        log("[Init] FPS (forced):", self.fps)

    def export_current_timeline(self):
        scene = bpy.context.scene

        # Force evaluation at our fixed FPS for dt; we still iterate over current timeline frames
        dt = 1.0 / float(self.fps)

        f0, f1 = scene.frame_start, scene.frame_end
        T = f1 - f0 + 1
        J = len(self.node_names)
        log(f"[Timeline] frames {f0}..{f1} (T={T})")

        # Allocate arrays
        rotations = np.zeros((T, J, 4), dtype=np.float32)             # local xyzw
        root_translation = np.zeros((T, 3), dtype=np.float32)         # world pelvis pos
        global_velocity = np.zeros((T, J, 3), dtype=np.float32)       # m/s
        global_angular_velocity = np.zeros((T, J, 3), dtype=np.float32)  # rad/s
        rigidbody_global_pos = np.zeros((T, J, 3), dtype=np.float32)  # world
        rigidbody_global_rot = np.zeros((T, J, 4), dtype=np.float32)  # world xyzw
        box_poses = np.zeros((T, 7), dtype=np.float32)                # world [tx,ty,tz, qx,qy,qz,qw]

        # Sample all frames
        for t, f in enumerate(range(f0, f1 + 1)):
            scene.frame_set(f)

            # Root translation (world) from pelvis
            rt, _ = obj_world_pose(self.pelvis)
            root_translation[t] = rt

            # Joint local rotations and global poses from Blender
            for i, n in enumerate(self.node_names):
                child = self.node_obj[n]
                parent = self.parent_obj[n]

                # local rotation (child relative to parent)
                q_local_wxyz = local_quat_from_world(child, parent)  # Blender (w,x,y,z)
                rotations[t, i] = _wxyz_to_xyzw(q_local_wxyz)        # store as xyzw

                # global pose (world)
                gt, gq_xyzw = obj_world_pose(child)
                rigidbody_global_pos[t, i] = gt
                rigidbody_global_rot[t, i] = gq_xyzw

            # Box pose (world)
            bt, bq = obj_world_pose(self.box_obj)
            box_poses[t, :3] = bt
            box_poses[t, 3:] = bq

        # Compute global linear velocity and angular velocity
        for t in range(1, T):
            # linear velocity from world positions
            global_velocity[t, :, :] = (rigidbody_global_pos[t, :, :] - rigidbody_global_pos[t - 1, :, :]) / dt
            # angular velocity from quaternion delta
            for j in range(J):
                q_prev = rigidbody_global_rot[t - 1, j, :]  # xyzw
                q_curr = rigidbody_global_rot[t, j, :]      # xyzw
                global_angular_velocity[t, j, :] = _quat_delta_angular_velocity(q_prev, q_curr, dt)

        # Build skeleton.local_translation from first-frame global positions
        # local = child_global - parent_global; root uses its global position
        first_global = rigidbody_global_pos[0]  # (J,3)
        local_translation = np.zeros((J, 3), dtype=np.float32)
        for i in range(J):
            pidx = self.parents[i]
            if pidx < 0:
                local_translation[i] = first_global[i]
            else:
                local_translation[i] = first_global[i] - first_global[pidx]

        skeleton_tree = {
            "node_names": self.node_names,
            "parent_indices": {"arr": self.parents},
            "local_translation": {"arr": local_translation},
        }

        # Pack reference motion dict in a TokenHSI-like layout
        ref_motion = {
            "__name__": "SkeletonMotion",
            "fps": int(self.fps),
            "is_local": True,
            "skeleton_tree": skeleton_tree,

            # local joint rotations (xyzw) and root translation from Blender
            "rotation": {"arr": rotations},
            "root_translation": {"arr": root_translation},

            # world-space measures sourced from Blender
            "global_translation": rigidbody_global_pos,     # (T,J,3)
            "global_rotation":    rigidbody_global_rot,     # (T,J,4) xyzw

            # derived kinematics
            "global_velocity": {"arr": global_velocity},                 # (T,J,3) m/s
            "global_angular_velocity": {"arr": global_angular_velocity}, # (T,J,3) rad/s
        }

        # Save
        ref_path = os.path.join(self.output_dir, "ref_motion.npy")
        box_path = os.path.join(self.output_dir, "box_motion.npy")
        np.save(ref_path, ref_motion)
        np.save(box_path, box_poses)
        log("[OK] Saved:", ref_path)
        log("[OK] Saved:", box_path)


# ---------------- Run ----------------------
if __name__ == "__main__":
    exporter = EmptiesIsaacExporter(
        node_names=NODE_NAMES,
        parents=PARENTS,
        pelvis_empty_name=PELVIS_EMPTY_NAME,
        box_object_name=BOX_OBJECT_NAME,
        fps=FPS,                         # fixed at 20
        output_dir=OUTPUT_DIR,           # //blender/
        humanoid_xml_path=HUMANOID_XML_PATH,
        do_scenepic=DO_SCENEPIC,
    )
    exporter.export_current_timeline()
    log("[DONE] Export complete.")