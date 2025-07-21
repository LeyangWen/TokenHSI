import torch

def load_pth(filepath: str):
    """
    Load a .pth file and return its state dict.
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    # If it’s a training checkpoint dict, extract the inner state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint

if __name__ == "__main__":
    # Point this to your checkpoint
    filepath = "/home/wenleyan/projects/TokenHSI/output/tokenhsi/ckpt_stage1.pth"
    state_dict = load_pth(filepath)

    print(f"Loaded {len(state_dict)} tensors from '{filepath}':\n")
    for key, tensor in state_dict.items():
        # some entries might not be tensors (e.g. optimizer states), guard against that
        if hasattr(tensor, "shape"):
            print(f"{key:60s}  →  {tuple(tensor.shape)}")
        else:
            print(f"{key:60s}  →  {type(tensor).__name__}")
