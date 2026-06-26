"""Run the stock line Phi-space clustering project."""

from pathlib import Path
import argparse

from src.stock_line_phi import ChartConfig, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Stock-like line chart pixel-grid clustering with Phi features.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where output files will be saved.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the generated chart.")
    parser.add_argument("--known-lines", type=int, default=2, help="Known number of chart lines to separate.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ChartConfig(seed=args.seed, k_known_lines=args.known_lines)
    paths = run_pipeline(output_dir=Path(args.output_dir), cfg=cfg)
    print("Pipeline complete.")
    print(f"Outputs written to: {paths['output_dir'].resolve()}")
    print(f"Feature table: {paths['phi_features_csv']}")
    print(f"Intersections: {paths['intersections_csv']}")
    print(f"Interactive 3D plot: {paths['interactive_3d']}")


if __name__ == "__main__":
    main()
