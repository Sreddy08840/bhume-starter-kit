"""
Test script for confidence calibration.
"""

from solution import (
    ConfidenceCalibrator,
    ConfidenceInputs,
)


def main():
    # Initialize calibrator
    calibrator = ConfidenceCalibrator()

    # Test case 1: All scores perfect
    print("--- Test 1: All perfect scores ---")
    inputs1 = ConfidenceInputs(
        alignment_score=1.0,
        area_score=1.0,
        edge_score=1.0,
        neighbor_score=1.0,
    )
    result1 = calibrator.calibrate(inputs1)
    print(f"Final confidence: {result1.confidence:.4f}")

    # Test case 2: All scores mediocre
    print("\n--- Test 2: All mediocre scores ---")
    inputs2 = ConfidenceInputs(
        alignment_score=0.5,
        area_score=0.5,
        edge_score=0.5,
        neighbor_score=0.5,
    )
    result2 = calibrator.calibrate(inputs2)
    print(f"Final confidence: {result2.confidence:.4f}")

    # Test case 3: One very low score (ambiguous alignment)
    print("\n--- Test 3: One very low score ---")
    inputs3 = ConfidenceInputs(
        alignment_score=0.1,
        area_score=0.8,
        edge_score=0.8,
        neighbor_score=0.8,
    )
    result3 = calibrator.calibrate(inputs3)
    print(f"Final confidence: {result3.confidence:.4f}")

    # Test case 4: Mixed scores (some good, some bad)
    print("\n--- Test 4: Mixed scores ---")
    inputs4 = ConfidenceInputs(
        alignment_score=0.9,
        area_score=0.3,
        edge_score=0.7,
        neighbor_score=0.6,
    )
    result4 = calibrator.calibrate(inputs4)
    print(f"Final confidence: {result4.confidence:.4f}")


if __name__ == "__main__":
    main()
