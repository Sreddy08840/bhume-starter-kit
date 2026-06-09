"""
Prediction generation module: Creates predictions in the required output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from bhume import write_predictions


@dataclass
class PlotPrediction:
    """
    Single plot prediction data.

    Attributes:
        plot_number: Unique plot identifier
        status: 'corrected' or 'flagged'
        confidence: Confidence value [0, 1]
        method_note: Note about the method used
        geometry: Corrected or original geometry
    """
    plot_number: str
    status: str
    confidence: float
    method_note: str
    geometry: BaseGeometry


class PredictionGenerator:
    """
    Generates predictions in the required schema.
    """

    def __init__(
        self,
        corrected_threshold: float = 0.7,
    ):
        """
        Initialize the prediction generator.

        Args:
            corrected_threshold: Confidence threshold for 'corrected' status
        """
        self.corrected_threshold = corrected_threshold

    def generate(
        self,
        plot_predictions: List[PlotPrediction],
    ) -> gpd.GeoDataFrame:
        """
        Generate predictions GeoDataFrame from plot predictions.

        Args:
            plot_predictions: List of PlotPrediction objects

        Returns:
            GeoDataFrame in the required schema
        """
        # Validate all predictions first
        self._validate_predictions(plot_predictions)

        # Create GeoDataFrame
        predictions_data = [
            {
                "plot_number": p.plot_number,
                "status": p.status,
                "confidence": p.confidence,
                "method_note": p.method_note,
                "geometry": p.geometry,
            }
            for p in plot_predictions
        ]

        predictions_gdf = gpd.GeoDataFrame(
            predictions_data,
            geometry="geometry",
            crs="EPSG:4326",
        )
        predictions_gdf = predictions_gdf.set_index("plot_number", drop=False)

        return predictions_gdf

    def create_plot_prediction(
        self,
        plot_number: str,
        confidence: float,
        corrected_geometry: BaseGeometry,
        original_geometry: BaseGeometry,
        method_note: str = "",
    ) -> PlotPrediction:
        """
        Create a single PlotPrediction object using the confidence threshold rule.

        Args:
            plot_number: Unique plot identifier
            confidence: Confidence value [0, 1]
            corrected_geometry: Corrected geometry to use if confidence is high
            original_geometry: Original geometry to use if confidence is low
            method_note: Note about the method used

        Returns:
            PlotPrediction object with correct status and geometry
        """
        confidence = max(0.0, min(1.0, confidence))
        if confidence > self.corrected_threshold:
            status = "corrected"
            geometry = corrected_geometry
        else:
            status = "flagged"
            geometry = original_geometry

        return PlotPrediction(
            plot_number=plot_number,
            status=status,
            confidence=confidence,
            method_note=method_note,
            geometry=geometry,
        )

    def save(
        self,
        predictions: gpd.GeoDataFrame,
        output_path: str | Path,
    ) -> Path:
        """
        Save predictions to a file.

        Args:
            predictions: Predictions GeoDataFrame
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        return write_predictions(output_path, predictions)

    def _validate_predictions(
        self,
        plot_predictions: List[PlotPrediction],
    ) -> None:
        """
        Validate a list of plot predictions.

        Args:
            plot_predictions: List of PlotPrediction objects

        Raises:
            ValueError: If any validation fails
        """
        plot_numbers = set()
        for pred in plot_predictions:
            # Check plot number
            if not isinstance(pred.plot_number, str) or not pred.plot_number:
                raise ValueError(f"Invalid plot_number: {pred.plot_number}")
            if pred.plot_number in plot_numbers:
                raise ValueError(f"Duplicate plot_number: {pred.plot_number}")
            plot_numbers.add(pred.plot_number)

            # Check status
            if pred.status not in ["corrected", "flagged"]:
                raise ValueError(f"Invalid status for plot {pred.plot_number}: {pred.status}")

            # Check confidence
            if not isinstance(pred.confidence, (int, float)):
                raise ValueError(f"Confidence must be numeric for plot {pred.plot_number}")
            if pred.confidence < 0.0 or pred.confidence > 1.0:
                raise ValueError(f"Confidence out of range [0,1] for plot {pred.plot_number}")

            # Check method note
            if not isinstance(pred.method_note, str):
                raise ValueError(f"method_note must be a string for plot {pred.plot_number}")

            # Check geometry
            if pred.geometry is None or pred.geometry.is_empty:
                raise ValueError(f"Invalid geometry for plot {pred.plot_number}")


# Import Path here to avoid circular imports
from pathlib import Path
