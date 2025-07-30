import torch
import torch.nn as nn

def l2_norm_calculation(model_a, model_b):
    """
    Computes the L2 norm of the difference between the weights of two HuggingFace models.
    Assumes both models have the same architecture.
    """
    norm = 0.0
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    for name in params_a:
        if name in params_b:
            diff = params_a[name].data - params_b[name].data
            norm += torch.norm(diff, p=2).item() ** 2
    return norm ** 0.5