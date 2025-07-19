import json

def load_mask(file_path: str) -> dict:
    """
    Load Wanda binary mask from a JSON file.
    Format: {"layer_0.ffn": 1, "layer_0.head_0": 0, ...}
    """
    with open(file_path, "r") as f:
        return json.load(f)
