"""
Complete end-to-end pipeline for cadastral boundary correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List
import geopandas as gpd

from .loader import VillageDataLoader, VillageData
from .patch_extraction import PatchExtractor, Patch
from .boundary_detection import FieldBoundaryDetector, BoundaryDetectionResult
from .alignment import PlotAligner, AlignmentResult
from .area_validation import AreaValidator, AreaValidationResult, AreaFlagReason
from .neighbor_consistency import NeighborConsistencyScorer, NeighborConsistencyResult, PlotShift
from .confidence_calibration import ConfidenceCalibrator, ConfidenceInputs, ConfidenceResult
from .prediction_generator import PredictionGenerator, PlotPrediction


class BoundaryCorrectionPipeline:
    """
    Complete end-to-end pipeline for correcting cadastral boundaries.
    """

    def __init__(
        self,
        corrected_threshold: float = 0.7,
    ):
        """
        Initialize the pipeline.

        Args:
            corrected_threshold: Confidence threshold for 'corrected' status
        """
        self.corrected_threshold = corrected_threshold

        # Initialize all components
        self.patch_extractor: Optional[PatchExtractor] = None
        self.boundary_detector = FieldBoundaryDetector()
        self.aligner = PlotAligner()
        self.area_validator = AreaValidator()
        self.neighbor_scorer = NeighborConsistencyScorer()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.prediction_generator = PredictionGenerator(corrected_threshold=corrected_threshold)

        # Store intermediate results
        self.plot_shifts: Dict[str, PlotShift] = {}
        self.original_geometries: Dict[str, BaseGeometry] = {}
        self.corrected_geometries: Dict[str, BaseGeometry] = {}
        self.confidence_components: Dict[str, ConfidenceInputs] = {}

    def run(
        self,
        village_dir: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> gpd.GeoDataFrame:
        """
        Run the complete pipeline on a village.

        Args:
            village_dir: Path to village data directory
            output_path: Optional path to save predictions.geojson

        Returns:
            Predictions GeoDataFrame
        """
        # Step 1: Load village data
        print(f"Loading village data from {village_dir}...")
        loader = VillageDataLoader(village_dir)
        village_data = loader.load()

        # Initialize patch extractor
        self.patch_extractor = PatchExtractor(
            imagery_path=village_data.imagery_path,
            boundaries_path=village_data.boundaries_path,
        )

        # Step 2: Process each plot to get initial shifts and components
        print(f"Processing {len(village_data.plots)} plots...")
        self.plot_shifts = {}
        self.original_geometries = {}
        self.corrected_geometries = {}
        self.confidence_components = {}

        for plot_num in village_data.plots.index:
            plot_geom = village_data.get_plot(plot_num)
            self.original_geometries[plot_num] = plot_geom
            self._process_plot(plot_num, plot_geom, village_data)

        # Step 3: Re-calculate neighbor scores with all shifts known and generate predictions
        print("Calculating final neighbor consistency scores and generating predictions...")
        plot_predictions: List[PlotPrediction] = []

        for plot_num in village_data.plots.index:
            plot_geom = self.original_geometries[plot_num]
            neighbors = village_data.find_neighbors(plot_num, distance_m=1.0)
            neighbor_shifts = [
                self.plot_shifts[n] for n in neighbors if n in self.plot_shifts
            ]

            plot_shift = self.plot_shifts[plot_num]
            neighbor_result = self.neighbor_scorer.score(plot_shift, neighbor_shifts)

            # Get confidence components and update neighbor score
            components = self.confidence_components[plot_num]
            final_components = ConfidenceInputs(
                alignment_score=components.alignment_score,
                area_score=components.area_score,
                edge_score=components.edge_score,
                neighbor_score=neighbor_result.neighbor_score,
            )

            # Calibrate final confidence
            final_confidence = self.confidence_calibrator.calibrate(final_components)

            # Create plot prediction
            plot_prediction = self.prediction_generator.create_plot_prediction(
                plot_number=plot_num,
                confidence=final_confidence.confidence,
                corrected_geometry=self.corrected_geometries[plot_num],
                original_geometry=plot_geom,
                method_note=f"dx={plot_shift.dx:.2f}m, dy={plot_shift.dy:.2f}m",
            )
            plot_predictions.append(plot_prediction)

        # Step 4: Generate predictions GeoDataFrame
        final_gdf = self.prediction_generator.generate(plot_predictions)

        # Step 5: Save output if requested
        if output_path is not None:
            print(f"Saving predictions to {output_path}...")
            self.prediction_generator.save(final_gdf, output_path)

        return final_gdf

    def _process_plot(
        self,
        plot_num: str,
        plot_geom,
        village_data: VillageData,
    ) -> None:
        """
        Process a single plot through the pipeline.
        """
        try:
            # 1. Extract patch
            patch: Patch = self.patch_extractor.extract(plot_geom, buffer_meters=30.0)

            # 2. Detect boundaries
            detection_result: BoundaryDetectionResult = self.boundary_detector.detect(
                patch.rgb_image, patch.boundary_mask
            )

            # 3. Align plot
            alignment_result: AlignmentResult = self.aligner.align(
                official_polygon=plot_geom,
                boundary_mask=detection_result.binary_mask,
                patch_transform=patch.transform,
                patch_crs=patch.crs,
            )

            # 4. Validate area
            recorded_area = village_data.calculate_area(plot_num)
            from .alignment import _utm_crs_for, _reproject_geometry
            utm_crs = _utm_crs_for(plot_geom.centroid.x, plot_geom.centroid.y)
            aligned_geom_utm = _reproject_geometry(alignment_result.best_polygon, "EPSG:4326", utm_crs)
            aligned_area = aligned_geom_utm.area
            area_result: AreaValidationResult = self.area_validator.validate(recorded_area, aligned_area)

            # 5. Store results
            self.plot_shifts[plot_num] = PlotShift(
                plot_number=plot_num,
                dx=alignment_result.dx,
                dy=alignment_result.dy,
            )
            self.corrected_geometries[plot_num] = alignment_result.best_polygon

            # 6. Prepare confidence components (neighbor score will be updated later)
            self.confidence_components[plot_num] = ConfidenceInputs(
                alignment_score=alignment_result.score,
                area_score=area_result.area_score,
                edge_score=detection_result.edge_confidence,
                neighbor_score=0.5,  # Neutral until all shifts are known
            )

        except Exception as e:
            # If anything fails, store original geometry and low confidence
            print(f"Warning: Failed to process plot {plot_num}: {e}")
            self.plot_shifts[plot_num] = PlotShift(plot_number=plot_num, dx=0.0, dy=0.0)
            self.corrected_geometries[plot_num] = plot_geom
            self.confidence_components[plot_num] = ConfidenceInputs(
                alignment_score=0.0,
                area_score=0.0,
                edge_score=0.0,
                neighbor_score=0.5,
            )
