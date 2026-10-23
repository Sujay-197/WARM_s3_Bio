"""Run WARL prediction pipeline from the command line."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warl.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("interaction_path")
    parser.add_argument("--cache-path", default="data/cache/go_annotations.json")
    parser.add_argument("--threshold", type=float, default=1e-14)
    args = parser.parse_args()
    predictions = run_pipeline(args.interaction_path, args.cache_path, args.threshold)
    sys.stdout.write(predictions.to_csv(index=False))


if __name__ == "__main__":
    main()
