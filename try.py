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
    filepath = "output/tokenhsi/ckpt_stage1.pth"
    state_dict_1 = load_pth(filepath)
    
    model_1 = state_dict_1["model"]
    
    state_dict_2 = load_pth("output/single_task/ckpt_carry.pth")
    model_2 = state_dict_2["model"]

    
    # Compare the two models
    for key in model_1.keys():
        if key not in model_2:
            print(f"Key '{key}' not found in model_2")
        else:
            if torch.equal(model_1[key], model_2[key]):
                print(f"Key '{key}' is equal in both models")
            else:
                print(f"Key '{key}' differs between models")
    for key in model_2.keys():
        if key not in model_1:
            print(f"Key '{key}' not found in model_1")
            
            
    print(f"Loaded {len(state_dict_1)} tensors from '{filepath}':\n")
    # for key, tensor in state_dict.items():
    #     # some entries might not be tensors (e.g. optimizer states), guard against that
    #     if hasattr(tensor, "shape"):
    #         print(f"{key:60s}  →  {tuple(tensor.shape)}")
    #     else:
    #         print(f"{key:60s}  →  {type(tensor).__name__}")
    
    
    # tensor.keys()
    
    model_2['self_encoder']
any_loaded = False
for k in model_2.keys():
    print(k)
        

        