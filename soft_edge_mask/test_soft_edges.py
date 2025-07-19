from label_smoother import compute_z_star
from edge_score_computer import compute_edge_scores

# Sample test case (hardcoded)
mock_mask = {
    "layer_0.ffn": 1,
    "layer_0.head_0": 0,
    "layer_0.head_1": 1,
    "layer_1.ffn": 0,
    "layer_1.head_0": 1,
    "layer_1.head_1": 0
}

z_star = compute_z_star(mock_mask, alpha=0.1)
edge_scores = compute_edge_scores(z_star)

for k, v in edge_scores.items():
    print(k, v)
