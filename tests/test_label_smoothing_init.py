import unittest
import torch
import json
import os
import sys

# Add the project's src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from layer1.soft_edge_mask.label_smoothing_utils import load_mask, compute_z_star, compute_edge_scores
from layer2.modeling.modeling_fpt2 import FPT2LMHeadModel

class TestLabelSmoothingInit(unittest.TestCase):

    def setUp(self):
        from transformers import GPT2Config
        config = GPT2Config(n_layer=2, n_head=2)
        self.model = FPT2LMHeadModel(config, with_embedding_nodes=True)
        self.model.eval()

        self.mask_path = os.path.join(os.path.dirname(__file__), "dummy_mask.json")
        # Use valid WRITER names as keys for the mask
        dummy_mask = {
            "tok_embeds": {"score": 1.0, "mask": 1},
            "a0.h0": {"score": 0.0, "mask": 0} 
        }
        with open(self.mask_path, "w") as f:
            json.dump(dummy_mask, f)

    def tearDown(self):
        os.remove(self.mask_path)

    def test_initialization(self):
        # We'll test the edge from 'tok_embeds' to the reader 'a0.h0.q'
        original_log_alpha = self.model.transformer.h[0].q_read_log_alphas[0, 0].item()

        alpha = 0.1
        mask = load_mask(self.mask_path)
        z_star = compute_z_star(mask, alpha=alpha)
        edge_scores = compute_edge_scores(z_star)
        self.model.initialize_alphas_from_scores(edge_scores)

        # The new value will be set for the Q, K, and V readers of a0.h0
        new_log_alpha = self.model.transformer.h[0].q_read_log_alphas[0, 0].item()

        self.assertNotEqual(original_log_alpha, new_log_alpha)

        # Calculate the score for the specific edge we are testing
        score = edge_scores['tok_embeds']['a0.h0']
        expected_log_alpha = 10.0 + (score - 0.5) * 5.0
        self.assertAlmostEqual(new_log_alpha, expected_log_alpha, places=4)

if __name__ == '__main__':
    unittest.main()