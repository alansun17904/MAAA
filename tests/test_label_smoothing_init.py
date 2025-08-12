import unittest
import torch
import json
import os
import sys

# Setup the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from layer1.soft_edge_mask.label_smoothing_utils import load_mask, compute_z_star, compute_edge_scores
from layer2.modeling.modeling_fpt2 import FPT2LMHeadModel

class TestLabelSmoothingInit(unittest.TestCase):

    def setUp(self):
        # We only need to create the mask file here.
        self.mask_path = os.path.join(os.path.dirname(__file__), "dummy_mask.json")
        dummy_mask = {
            "tok_embeds": {"score": 1.0, "mask": 1},
            "a0.h0": {"score": 0.0, "mask": 0} 
        }
        with open(self.mask_path, "w") as f:
            json.dump(dummy_mask, f)

    def tearDown(self):
        os.remove(self.mask_path)

    def test_initialization(self):
        # 1. First, compute the scores from the mask
        alpha = 0.1
        mask = load_mask(self.mask_path)
        z_star = compute_z_star(mask, alpha=alpha)
        edge_scores = compute_edge_scores(z_star)

        # 2. Create the model configuration
        from transformers import GPT2Config
        config = GPT2Config(n_layer=2, n_head=2, vocab_size=100, n_embd=4) # Use small dimensions

        # 3. Create a NEW model instance, passing the scores DIRECTLY to the constructor
        initialized_model = FPT2LMHeadModel(
            config, 
            with_embedding_nodes=True, 
            initial_scores=edge_scores
        )

        # 4. Check the initialized value
        # This checks the edge from 'tok_embeds' (writer 0) to head 0 of layer 0.
        new_log_alpha = initialized_model.transformer.h[0].q_read_log_alphas[0, 0].item()

        # 5. Calculate the expected value for the assertion
        score = edge_scores['tok_embeds']['a0.h0']
        expected_log_alpha = 10.0 + (score - 0.5) * 5.0
        
        # 6. Assert that the values are almost equal
        self.assertAlmostEqual(new_log_alpha, expected_log_alpha, places=4)

if __name__ == '__main__':
    unittest.main()