"""
Confidence calibration module: combines multiple scores into final confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConfidenceInputs:
    """
    Input scores for confidence calibration.

    Attributes:
        alignment_score: Score from plot alignment [0, 1]
        area_score: Score from area validation [0, 1]
        edge_score: Score from edge detection confidence [0, 1]
        neighbor_score: Score from neighbor consistency [0, 1]
    """
    alignment_score: float
    area_score: float
    edge_score: float
    neighbor_score: float


@dataclass
class ConfidenceResult:
    """
    Result of confidence calibration.

    Attributes:
        confidence: Final calibrated confidence [0, 1]
        component_scores: Dictionary of individual component scores
    """
    confidence: float
    component_scores: ConfidenceInputs


class ConfidenceCalibrator:
    """
    Combines multiple quality scores into a single calibrated confidence value.
    """

    def __init__(
        self,
        alignment_weight: float = 0.35,
        area_weight: float = 0.25,
        edge_weight: float = 0.25,
        neighbor_weight: float = 0.15,
    ):
        """
        Initialize the confidence calibrator with component weights.

        Args:
            alignment_weight: Weight for alignment score
            area_weight: Weight for area score
            edge_weight: Weight for edge confidence
            neighbor_weight: Weight for neighbor consistency
        """
        self.alignment_weight = alignment_weight
        self.area_weight = area_weight
        self.edge_weight = edge_weight
        self.neighbor_weight = neighbor_weight

        # Normalize weights to sum to 1
        total = sum([alignment_weight, area_weight, edge_weight, neighbor_weight])
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        self.alignment_weight /= total
        self.area_weight /= total
        self.edge_weight /= total
        self.neighbor_weight /= total

    def calibrate(
        self,
        inputs: ConfidenceInputs,
    ) -> ConfidenceResult:
        """
        Calibrate confidence from input scores.

        Args:
            inputs: Object containing all four component scores

        Returns:
            ConfidenceResult with final confidence and component scores
        """
        # Clip all scores to [0, 1]
        alignment_score = np.clip(inputs.alignment_score, 0.0, 1.0)
        area_score = np.clip(inputs.area_score, 0.0, 1.0)
        edge_score = np.clip(inputs.edge_score, 0.0, 1.0)
        neighbor_score = np.clip(inputs.neighbor_score, 0.0, 1.0)

        # Calculate weighted average
        weighted_score = (
            self.alignment_weight * alignment_score +
            self.area_weight * area_score +
            self.edge_weight * edge_score +
            self.neighbor_weight * neighbor_score
        )

        # Calculate minimum component: penalize if any single score is very low
        min_score = min(alignment_score, area_score, edge_score, neighbor_score)

        # Combine weighted average * (boosted by consistency of all components being high
        confidence = float(weighted_score * (0.7 + 0.3 * min_score))

        # Clip to [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)

        return ConfidenceResult(
            confidence=confidence,
            component_scores=ConfidenceInputs(
                alignment_score=alignment_score,
                area_score=area_score,
                edge_score=edge_score,
                neighbor_score=neighbor_score,
            ),
        )
