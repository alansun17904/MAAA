import math
import json

def compute_z_star(pruneinfo: list, alpha: float = 0.1) -> dict:
    """
    Compute z_i* = 1 - alpha if retained (mask==1), else alpha if pruned (mask==0).
    Input: {"block.0.attn.W_0": {"score": ..., "mask": 1 or 0}, ...}
    Output: {"block.0.attn.W_0": 0.9, ...}
    """
    return {
        f"{info['name']}[{info['head_idx']}]" if "mlp" not in info['name'] else info['name']: (1 - alpha if info["mask"] == 1 else alpha)
        for info in pruneinfo
    }
def compute_edge_scores(z_star: dict) -> dict:
    """
    Compute P(z_i → z_j) = sqrt(z_i* * z_j*) for all component pairs.
    Returns a nested dictionary of scores: {comp_i: {comp_j: score, ...}, ...}
    """
    components = list(z_star.keys())
    edge_scores = {}
    writers = {}
    for comp_i in components:
        if "W_O" in comp_i or "W_out" in comp_i:
            writers[comp_i] = z_star[comp_i]
            continue
        layeri = int(comp_i.split(".")[1])
        name = f"{comp_i.split('.')[2]}.{comp_i.split('.')[-1]}"
        edge_scores[comp_i] = {}
        for comp_j in components: #Really inefficient frn but whatever
            layerj = int(comp_j.split(".")[1])
            name = f"{comp_j.split('.')[2]}.{comp_i.split('.')[-1]}"
            if (comp_i == comp_j) or (layerj >= layeri) or ("W_out" not in comp_j and ("W_O" not in comp_j)):
                continue  # skips if not writer
            
            score = math.sqrt(z_star[comp_i] * z_star[comp_j])
            edge_scores[comp_i][comp_j] = round(score, 4)
    return edge_scores, writers

def load_mask(filepath: str) -> dict:
    """
    This is for loading the new Wanda pruning mask (Aaron's output) :
    {
      "block.0.attn.W_0": {"score": 1.234, "mask": 1},
      "block.0.attn.W_1": {"score": 0.823, "mask": 0},
      ...
    }
    """
    with open(filepath, "r") as f:
        return json.load(f)
