import argparse
import json

from label_smoothing_utils import load_mask, compute_z_star, compute_edge_scores

def main():
    parser = argparse.ArgumentParser(description="Generate soft edge scores from a Wanda pruning mask.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Label smoothing alpha value.")
    parser.add_argument("--input", type=str, required=True, help="Path to the binary pruning mask (JSON).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the soft edge scores (JSON).")
    args = parser.parse_args()

    print(f"Alpha: {args.alpha}")
    print(f"Loading pruning mask from: {args.input}")

    # Step 1: Load binary mask
    try:
        mask = load_mask(args.input)
    except Exception as e:
        print(f"Error loading mask: {e}")
        return

    # Step 2: Compute smoothed importance scores
    try:
        z_star = compute_z_star(mask, alpha=args.alpha)
    except Exception as e:
        print(f"Error computing z_star: {e}")
        return

    # Step 3: Compute soft edge scores
    try:
        edge_scores = compute_edge_scores(z_star)
    except Exception as e:
        print(f"Error computing edge scores: {e}")
        return

    # Step 4: Save to output file
    try:
        with open(args.output, "w") as f:
            json.dump(edge_scores, f, indent=2)
        print(f"Soft edge scores written to: {args.output}")
    except Exception as e:
        print(f"Error saving edge scores: {e}")

if __name__ == "__main__":
    main()