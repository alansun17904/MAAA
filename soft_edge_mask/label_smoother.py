def compute_z_star(mask: dict, alpha: float = 0.1) -> dict:
    """
    Compute z_i* = 1 - alpha if retained (z_i = 1), else alpha if pruned (z_i = 0).
    """
    return {k: (1 - alpha if v == 1 else alpha) for k, v in mask.items()}
