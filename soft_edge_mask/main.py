import argparse
import json

from mask_loader import load_mask
from label_smoother import compute_z_star
from edge_score_computer import compute_edge_scores

def main():
    parser = argparse.ArgumentParser(description="Generate soft edge scores from Wanda pruning mask")
    parser.add_argument("--alpha", type=float, default=0.1, help="Label smoothing alpha value")
    parser.add_argument("--input", type=str, required=True, help="Path to Wanda binary pruning mask (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Path to save soft edge scores (JSON)")
    args = parser.parse_args()

    print(f"\n🔧 Using alpha = {args.alpha}")
    print(f"📂 Loading mask from: {args.input}")

    # Step 1: Load mask
    try:
        mask = load_mask(args.input)
        print("✅ Loaded mask:")
        print(mask)
    except Exception as e:
        print("❌ Failed to load mask:", e)
        return

    # Step 2: Compute z_star
    try:
        z_star = compute_z_star(mask, alpha=args.alpha)
        print("✅ z* values:")
        print(z_star)
    except Exception as e:
        print("❌ Failed to compute z_star:", e)
        return

    # Step 3: Compute edge scores
    try:
        edge_scores = compute_edge_scores(z_star)
        print("✅ Soft edge scores:")
        print(json.dumps(edge_scores, indent=2))
    except Exception as e:
        print("❌ Failed to compute edge scores:", e)
        return

    # Step 4: Save to file
    try:
        with open(args.output, "w") as f:
            json.dump(edge_scores, f, indent=2)
        print(f"💾 Saved soft edge scores to {args.output}")
    except Exception as e:
        print("❌ Failed to save edge scores:", e)

if __name__ == "__main__":
    main()
