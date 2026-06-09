"""
Area validation module: checks polygon area against recorded area.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AreaFlagReason(Enum):
    """Reasons for flagging an area as suspicious."""
    OK = "OK"
    AREA_TOO_SMALL = "Area too small"
    AREA_TOO_LARGE = "Area too large"
    INVALID_AREA = "Invalid area (zero or negative)"


@dataclass
class AreaValidationResult:
    """
    Result of area validation.

    Attributes:
        area_ratio: Ratio of polygon area to recorded area (polygon_area / recorded_area)
        area_score: Quality score [0, 1] (higher is better)
        flag_reason: Reason for flag (or OK if no flag)
        is_suspicious: Whether the area is suspicious
    """
    area_ratio: float
    area_score: float
    flag_reason: AreaFlagReason
    is_suspicious: bool


class AreaValidator:
    """
    Validates polygon area against recorded area.
    """

    def __init__(
        self,
        min_ratio: float = 0.5,
        max_ratio: float = 2.0,
    ):
        """
        Initialize the area validator.

        Args:
            min_ratio: Minimum allowed area ratio (polygon_area / recorded_area)
            max_ratio: Maximum allowed area ratio
        """
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def validate(
        self,
        recorded_area: float,
        polygon_area: float,
    ) -> AreaValidationResult:
        """
        Validate polygon area against recorded area.

        Args:
            recorded_area: Recorded area from cadastre (same units as polygon_area)
            polygon_area: Calculated area of polygon

        Returns:
            AreaValidationResult with ratio, score, flag reason, and suspicious flag
        """
        # Handle invalid areas
        if recorded_area <= 0 or polygon_area <= 0:
            return AreaValidationResult(
                area_ratio=0.0,
                area_score=0.0,
                flag_reason=AreaFlagReason.INVALID_AREA,
                is_suspicious=True,
            )

        # Calculate area ratio
        area_ratio = polygon_area / recorded_area

        # Calculate score
        area_score = self._calculate_score(area_ratio)

        # Determine flag reason
        flag_reason = AreaFlagReason.OK
        is_suspicious = False

        if area_ratio < self.min_ratio:
            flag_reason = AreaFlagReason.AREA_TOO_SMALL
            is_suspicious = True
        elif area_ratio > self.max_ratio:
            flag_reason = AreaFlagReason.AREA_TOO_LARGE
            is_suspicious = True

        return AreaValidationResult(
            area_ratio=area_ratio,
            area_score=area_score,
            flag_reason=flag_reason,
            is_suspicious=is_suspicious,
        )

    def _calculate_score(self, area_ratio: float) -> float:
        """
        Calculate area quality score.

        Args:
            area_ratio: Ratio of polygon area to recorded area

        Returns:
            Score in [0, 1] where 1 is perfect match
        """
        # Score is highest when ratio is 1
        # Symmetric around 1
        if area_ratio <= 0:
            return 0.0

        # Calculate distance from 1
        distance = abs(1.0 - area_ratio)

        # Use exponential decay for score
        # Score = exp(-k * distance), where k controls steepness
        k = 3.0
        score = float(max(0.0, min(1.0, np.exp(-k * distance))))

        return score


# Import numpy here to avoid circular imports
import numpy as np
