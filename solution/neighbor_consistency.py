"""
Neighbor consistency scoring module: checks if a plot's shift is consistent with neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np


@dataclass
class PlotShift:
    """
    Shift information for a single plot.

    Attributes:
        plot_number: Unique plot identifier
        dx: Translation in x-direction (meters)
        dy: Translation in y-direction (meters)
    """
    plot_number: str
    dx: float
    dy: float


@dataclass
class NeighborConsistencyResult:
    """
    Result of neighbor consistency check.

    Attributes:
        neighbor_score: Consistency score [0, 1] (higher is more consistent)
        neighbor_shifts: List of shifts for adjacent plots
        direction_similarity: Average cosine similarity of shift directions
        magnitude_similarity: Average similarity of shift magnitudes
    """
    neighbor_score: float
    neighbor_shifts: List[PlotShift]
    direction_similarity: float
    magnitude_similarity: float


class NeighborConsistencyScorer:
    """
    Scores the consistency of a plot's shift with its neighbors.
    """

    def __init__(
        self,
        direction_weight: float = 0.6,
        magnitude_weight: float = 0.4,
    ):
        """
        Initialize the neighbor consistency scorer.

        Args:
            direction_weight: Weight for direction similarity in final score
            magnitude_weight: Weight for magnitude similarity in final score
        """
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight

    def score(
        self,
        plot_shift: PlotShift,
        neighbor_shifts: List[PlotShift],
    ) -> NeighborConsistencyResult:
        """
        Calculate neighbor consistency score.

        Args:
            plot_shift: Shift for the plot to evaluate
            neighbor_shifts: Shifts for adjacent plots

        Returns:
            NeighborConsistencyResult with score and details
        """
        if not neighbor_shifts:
            # No neighbors - return neutral score
            return NeighborConsistencyResult(
                neighbor_score=0.5,
                neighbor_shifts=[],
                direction_similarity=0.5,
                magnitude_similarity=0.5,
            )

        # Calculate direction similarities
        direction_similarities = []
        plot_vec = np.array([plot_shift.dx, plot_shift.dy])
        plot_mag = np.linalg.norm(plot_vec)

        for neighbor in neighbor_shifts:
            neighbor_vec = np.array([neighbor.dx, neighbor.dy])
            neighbor_mag = np.linalg.norm(neighbor_vec)

            # Cosine similarity for direction
            if plot_mag < 1e-8 or neighbor_mag < 1e-8:
                # Both vectors are near zero - direction is same (no shift)
                dir_sim = 1.0
            else:
                dir_sim = float(np.dot(plot_vec, neighbor_vec) / (plot_mag * neighbor_mag))
                # Ensure similarity is in [-1, 1]
                dir_sim = max(-1.0, min(1.0, dir_sim))
            direction_similarities.append(dir_sim)

        # Calculate magnitude similarities
        magnitude_similarities = []
        for neighbor in neighbor_shifts:
            neighbor_vec = np.array([neighbor.dx, neighbor.dy])
            neighbor_mag = np.linalg.norm(neighbor_vec)

            # Magnitude similarity: 1 - |(mag1 - mag2)| / max(mag1, mag2, eps)
            max_mag = max(plot_mag, neighbor_mag, 1e-8)
            mag_diff = abs(plot_mag - neighbor_mag)
            mag_sim = 1.0 - (mag_diff / max_mag)
            mag_sim = max(0.0, min(1.0, mag_sim))
            magnitude_similarities.append(mag_sim)

        # Average similarities
        avg_dir_sim = float(np.mean(direction_similarities))
        avg_mag_sim = float(np.mean(magnitude_similarities))

        # Normalize direction similarity to [0, 1] (from [-1, 1])
        normalized_dir_sim = (avg_dir_sim + 1.0) / 2.0

        # Calculate final score
        neighbor_score = (
            self.direction_weight * normalized_dir_sim +
            self.magnitude_weight * avg_mag_sim
        )

        return NeighborConsistencyResult(
            neighbor_score=neighbor_score,
            neighbor_shifts=neighbor_shifts,
            direction_similarity=avg_dir_sim,
            magnitude_similarity=avg_mag_sim,
        )
