import math

def compute_edge_scores(z_star: dict) -> dict:
    """
    Compute P(z_i → z_j) = sqrt(z_i* * z_j*) for all component pairs.
    Returns a nested dictionary of scores.
    """
    components = list(z_star.keys())
    edge_scores = {}

    for comp_i in components:
        edge_scores[comp_i] = {}
        for comp_j in components:
            if comp_i == comp_j:
                continue  # optional: skip self-loops
            score = math.sqrt(z_star[comp_i] * z_star[comp_j])
            edge_scores[comp_i][comp_j] = round(score, 4)
    return edge_scores
