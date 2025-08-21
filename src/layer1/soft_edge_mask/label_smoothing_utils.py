import math
import json

def compute_z_star(pruneinfo: list, alpha: float = 0.1) -> dict:
    """
    Compute z_i* = 1 - alpha if retained (mask==1), else alpha if pruned (mask==0).
    Input: {"block.0.attn.W_0": {"score": ..., "mask": 1 or 0}, ...}
    Output: {"block.0.attn.W_0": 0.9, ...}
    """
    return {
        f"{info['name']}[{info['head_idx']}]" if "mlp" not in info['name'] else info['name']  : (1 - alpha if info["mask"] == 1 else alpha)
        for info in pruneinfo
    }
def writer_name_to_idx(name, layer_idx, head_idx, num_layers, num_heads, with_embedding_nodes=False):
    idx = 0
    if with_embedding_nodes:
        if name == "tok_embeds":
            return 0
        elif name == "pos_embeds":
            return 1
        else:
            idx += 2
    if "W_out" in name:
        idx += layer_idx * (num_heads + 1) + num_heads
    elif "W_O" in name:
        idx += layer_idx * (num_heads + 1) + head_idx
    else:
        raise ValueError(f"Unrecognized writer name {name}")
    return idx
def get_num_writers(layer, with_embedding_nodes=False):
            # If we include embedding nodes, there should be two for inputs_embeds and pos_embeds
            n_writers = 2 if with_embedding_nodes else 0
            n_writers += layer * (13)   # Each head's O and the MLP
            return n_writers
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
            head = 0
            name = comp_i
            if "W_out" not in comp_i:
                name = comp_i.split("[")[0]
                head = int(comp_i.split("[")[-1].split("]")[0])
            if name not in writers:
                writers[name] = [None]
                if "mlp" not in comp_i:
                    writers[name].extend([None] * 11)
            writers[name][head] = z_star[comp_i]
            continue
        layeri = int(comp_i.split(".")[1])
        head = 0
        name = comp_i
        if "W_in" not in comp_i:
            name = comp_i.split("[")[0]
            
            head = int(comp_i.split("[")[-1].split("]")[0])
        if name not in edge_scores:
            edge_scores[name] = [None]
            
            if "mlp" not in comp_i:
                edge_scores[name].extend([None] * 11)
        
        temp = []
        
        temp.extend([None] * get_num_writers(layeri))
        for comp_j in components: #Really inefficient frn but whatever
            layerj = int(comp_j.split(".")[1])
            headj = 0
            namej = comp_j
            
            if (comp_i == comp_j) or (layerj >= layeri) or ("W_out" not in comp_j and ("W_O" not in comp_j)):
                continue  # skips if not writer
            
            if "W_out" not in comp_j:
                namej = comp_j.split("[")[0]
                headj = int(comp_j.split("[")[-1].split("]")[0])
            '''
            if comp_i == "block.1.attn.W_Q[5]":
                print(comp_j)
                print(writer_name_to_idx(namej, layerj, headj, 12, 12))
            '''
            score = round(math.sqrt(z_star[comp_i] * z_star[comp_j]), 4)
            temp[writer_name_to_idx(namej, layerj, headj, 12, 12)] = score
        edge_scores[name][head] = temp
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
