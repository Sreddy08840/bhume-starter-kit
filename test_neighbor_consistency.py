"""
Test script for neighbor consistency scoring.
"""

from solution import (
    NeighborConsistencyScorer,
    PlotShift,
)


def main():
    # Test case 1: Perfectly consistent neighbors
    print("--- Test 1: Perfectly consistent neighbors ---")
    plot_shift = PlotShift("plot1", dx=5.0, dy=5.0)
    neighbor_shifts = [
        PlotShift("neighbor1", dx=5.0, dy=5.0),
        PlotShift("neighbor2", dx=5.2, dy=4.8),
        PlotShift("neighbor3", dx=4.8, dy=5.2),
    ]
    scorer = NeighborConsistencyScorer()
    result = scorer.score(plot_shift, neighbor_shifts)
    print(f"Neighbor score: {result.neighbor_score:.4f}")
    print(f"Direction similarity: {result.direction_similarity:.4f}")
    print(f"Magnitude similarity: {result.magnitude_similarity:.4f}")

    # Test case 2: Inconsistent neighbors
    print("\n--- Test 2: Inconsistent neighbors ---")
    plot_shift2 = PlotShift("plot1", dx=5.0, dy=5.0)
    neighbor_shifts2 = [
        PlotShift("neighbor1", dx=-5.0, dy=-5.0),  # Opposite direction
        PlotShift("neighbor2", dx=0.0, dy=0.0),   # No shift
        PlotShift("neighbor3", dx=20.0, dy=20.0), # Large magnitude
    ]
    result2 = scorer.score(plot_shift2, neighbor_shifts2)
    print(f"Neighbor score: {result2.neighbor_score:.4f}")
    print(f"Direction similarity: {result2.direction_similarity:.4f}")
    print(f"Magnitude similarity: {result2.magnitude_similarity:.4f}")

    # Test case 3: No neighbors
    print("\n--- Test 3: No neighbors ---")
    plot_shift3 = PlotShift("plot1", dx=5.0, dy=5.0)
    result3 = scorer.score(plot_shift3, [])
    print(f"Neighbor score: {result3.neighbor_score:.4f}")
    print(f"Direction similarity: {result3.direction_similarity:.4f}")
    print(f"Magnitude similarity: {result3.magnitude_similarity:.4f}")


if __name__ == "__main__":
    main()
