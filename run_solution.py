#!/usr/bin/env python3
"""
Main script to run the complete boundary correction solution.
"""

import sys
from pathlib import Path

from solution import BoundaryCorrectionPipeline
from bhume import load, score


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run run_solution.py <village_directory> [output_path]")
        print("Example: uv run run_solution.py data/34855_vadnerbhairav_chandavad_nashik")
        sys.exit(1)

    village_dir = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else village_dir / "predictions.geojson"

    # Run the pipeline
    pipeline = BoundaryCorrectionPipeline(confidence_threshold=0.3)
    predictions = pipeline.run(village_dir, output_path=output_path)

    print(f"\nProcessed {len(predictions)} plots")
    print(f"  Corrected: {len(predictions[predictions['status'] == 'corrected'])}")
    print(f"  Flagged: {len(predictions[predictions['status'] == 'flagged'])}")

    # Try to score against example truths if available
    try:
        village_data = load(village_dir)
        if village_data.example_truths is not None:
            print("\nScoring predictions against example truths...")
            score_result = score(predictions, village_data)
            print(score_result)
    except Exception as e:
        print(f"\nNote: Could not score predictions: {e}")


if __name__ == "__main__":
    main()
