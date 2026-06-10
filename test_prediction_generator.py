"""
Test script for prediction generator.
"""

import geopandas as gpd
from shapely.geometry import Polygon

from solution import PredictionGenerator, PlotPrediction


def main():
    print("Testing PredictionGenerator...")

    # Create a dummy plot geometry
    geom = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    # Initialize prediction generator with threshold 0.7
    generator = PredictionGenerator(corrected_threshold=0.7)

    # Create test predictions
    test_predictions = []

    # 1. High confidence, should be corrected
    pred1 = generator.create_plot_prediction(
        plot_number="plot_001",
        confidence=0.9,
        corrected_geometry=Polygon([(0.1, 0.1), (0.1, 1.1), (1.1, 1.1), (1.1, 0.1)]),
        original_geometry=geom,
        method_note="Global shift",
    )
    test_predictions.append(pred1)

    # 2. Low confidence, should be flagged
    pred2 = generator.create_plot_prediction(
        plot_number="plot_002",
        confidence=0.5,
        corrected_geometry=Polygon([(0.2, 0.2), (0.2, 1.2), (1.2, 1.2), (1.2, 0.2)]),
        original_geometry=geom,
        method_note="Edge detection weak",
    )
    test_predictions.append(pred2)

    # Generate GeoDataFrame
    predictions_gdf = generator.generate(test_predictions)
    print("Generated predictions GeoDataFrame:")
    print(predictions_gdf[["plot_number", "status", "confidence", "method_note"]])

    # Check results
    assert len(predictions_gdf) == 2
    assert predictions_gdf.loc["plot_001", "status"] == "corrected"
    assert predictions_gdf.loc["plot_002", "status"] == "flagged"
    assert 0 <= predictions_gdf.loc["plot_001", "confidence"] <= 1
    assert 0 <= predictions_gdf.loc["plot_002", "confidence"] <= 1

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
