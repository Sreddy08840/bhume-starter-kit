"""
Solution module for cadastral boundary correction.
"""

from .loader import VillageDataLoader
from .patch_extraction import PatchExtractor, Patch
from .boundary_detection import FieldBoundaryDetector, BoundaryDetectionResult
from .alignment import PlotAligner, AlignmentResult
from .area_validation import AreaValidator, AreaValidationResult, AreaFlagReason
from .neighbor_consistency import (
    NeighborConsistencyScorer,
    NeighborConsistencyResult,
    PlotShift,
)
from .confidence_calibration import (
    ConfidenceCalibrator,
    ConfidenceInputs,
    ConfidenceResult,
)
from .prediction_generator import PredictionGenerator, PlotPrediction
from .pipeline import BoundaryCorrectionPipeline

__all__ = [
    'VillageDataLoader',
    'PatchExtractor',
    'Patch',
    'FieldBoundaryDetector',
    'BoundaryDetectionResult',
    'PlotAligner',
    'AlignmentResult',
    'AreaValidator',
    'AreaValidationResult',
    'AreaFlagReason',
    'NeighborConsistencyScorer',
    'NeighborConsistencyResult',
    'PlotShift',
    'ConfidenceCalibrator',
    'ConfidenceInputs',
    'ConfidenceResult',
    'PredictionGenerator',
    'PlotPrediction',
    'BoundaryCorrectionPipeline',
]

