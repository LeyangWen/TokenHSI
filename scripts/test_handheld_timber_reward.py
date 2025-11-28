import os
import sys
from isaacgym.torch_utils import quat_rotate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENHSI_ROOT = os.path.join(REPO_ROOT, "tokenhsi")
for path in (REPO_ROOT, TOKENHSI_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from tokenhsi.env.tasks.basic_interaction_skills.humanoid_carry import (
    compute_handheld_timber_reward,
)

import torch


def build_test_batch():
    scenarios = [
        {
            "name": "ideal_grip",
            "description": "Hands straddle the board, 20cm from center, correct height.",
            "left_hand": (-0.20, 0.05, 0.95),
            "right_hand": (0.20, -0.05, 0.95),
            "box_pos": (0.0, 0.0, 1.0),
            "box_rot": (0.0, 0.0, 0.0, 1.0),
            "box_size": (1.8, 0.095, 0.045),
        },
        {
            "name": "same_side",
            "description": "Both hands on the same edge despite right spacing.",
            "left_hand": (0.20, 0.05, 0.95),
            "right_hand": (0.10, 0.05, 0.95),
            "box_pos": (0.0, 0.0, 1.0),
            "box_rot": (0.0, 0.0, 0.0, 1.0),
            "box_size": (1.8, 0.095, 0.045),
        },
        {
            "name": "tight_spacing",
            "description": "Hands too close to the center.",
            "left_hand": (-0.05, -0.04, 0.95),
            "right_hand": (0.05, 0.04, 0.95),
            "box_pos": (0.0, 0.0, 1.0),
            "box_rot": (0.0, 0.0, 0.0, 1.0),
            "box_size": (1.8, 0.095, 0.045),
        },
        {
            "name": "high_hands",
            "description": "Hands are well placed horizontally but lifted 10cm too high.",
            "left_hand": (-0.20, 0.05, 1.05),
            "right_hand": (0.20, -0.05, 1.05),
            "box_pos": (0.0, 0.0, 1.0),
            "box_rot": (0.0, 0.0, 0.0, 1.0),
            "box_size": (1.8, 0.095, 0.045),
        },
        {
            "name": "far_from_actor",
            "description": "Box is far from the humanoid root so reward should zero out.",
            "left_hand": (-0.20, 0.05, 0.95),
            "right_hand": (0.20, -0.05, 0.95),
            "box_pos": (2.0, 2.0, 1.0),
            "box_rot": (0.0, 0.0, 0.0, 1.0),
            "box_size": (1.8, 0.095, 0.045),
        },
        {
            "name": "rotated_board",
            "description": "Board yawed 45 deg; hands aligned with rotated ends.",
            "left_hand": (-0.15, -0.15, 0.95),
            "right_hand": (0.15, 0.15, 0.95),
            "box_pos": (0.0, 0.0, 1.0),
            "box_rot": (0.0, 0.0, 0.38268343, 0.92387953),
            "box_size": (1.8, 0.095, 0.045),
        },
    ]

    num_envs = len(scenarios)
    num_bodies = 3  # [root, left_hand, right_hand]

    humanoid = torch.zeros(num_envs, num_bodies, 3)
    humanoid[:, 0, :] = torch.tensor([0.0, 0.0, 1.0])  # root at origin
    box_pos = torch.zeros(num_envs, 3)
    box_size = torch.zeros(num_envs, 3)
    box_rot = torch.zeros(num_envs, 4)

    for idx, scenario in enumerate(scenarios):
        humanoid[idx, 1, :] = torch.tensor(scenario["left_hand"])
        humanoid[idx, 2, :] = torch.tensor(scenario["right_hand"])
        box_pos[idx, :] = torch.tensor(scenario["box_pos"])
        box_size[idx, :] = torch.tensor(scenario["box_size"])
        box_rot[idx, :] = torch.tensor(scenario["box_rot"])
    hands_ids = torch.tensor([1, 2], dtype=torch.long)

    return scenarios, humanoid, box_pos, box_rot, box_size, hands_ids




def main():
    scenarios, humanoid, box_pos, box_rot, box_size, hands_ids = build_test_batch()
    print("\nHandheld timber reward sanity check\n" + "-" * 40)
    for idx, scenario in enumerate(scenarios):
        print(f"Scenario: {scenario['name']}\n  {scenario['description']}")
        reward = compute_handheld_timber_reward(
            humanoid[idx : idx + 1],
            box_pos[idx : idx + 1],
            box_rot[idx : idx + 1],
            box_size[idx : idx + 1],
            hands_ids,
        )
        print(f"  reward: {reward[0].item():.3f}\n")
    

    print("Run `python scripts/test_handheld_timber_reward.py` to repeat this check.")


if __name__ == "__main__":
    main()
