"""
Test script for field boundary detection.
"""

from pathlib import Path
from PIL import Image
import numpy as np
from solution import (
    VillageDataLoader,
    PatchExtractor,
    FieldBoundaryDetector,
)


def main():
    # Replace with your village directory path
    village_dir = Path("data") / "34855_vadnerbhairav_chandavad_nashik"

    if not village_dir.exists():
        print(f"Village directory not found: {village_dir}")
        print("Please download a village bundle and place it in data/")
        return

    print(f"Loading village: {village_dir.name}")

    # Load village data
    loader = VillageDataLoader(village_dir)
    village_data = loader.load()

    # Get a plot
    plot_num = village_data.plots.index[0]
    plot_geom = village_data.get_plot(plot_num)
    print(f"Using plot: {plot_num}")

    # Extract patch
    patch_extractor = PatchExtractor(
        imagery_path=village_data.imagery_path,
        boundaries_path=village_data.boundaries_path
    )
    patch = patch_extractor.extract(plot_geom, buffer_meters=30.0)
    print(f"Extracted patch shape: {patch.rgb_image.shape}")

    # Detect boundaries
    detector = FieldBoundaryDetector()
    result = detector.detect(patch.rgb_image, patch.boundary_mask)

    print(f"\nBoundary detection complete!")
    print(f"Edge confidence: {result.edge_confidence:.4f}")
    print(f"Number of contours: {len(result.contours)}")

    # Save outputs
    print("\nSaving outputs...")

    # Save binary mask
    mask_image = (result.binary_mask * 255).astype(np.uint8)
    Image.fromarray(mask_image).save("boundary_mask.png")

    # Save overlay on original image
    overlay = patch.rgb_image.copy()
    overlay[result.binary_mask > 0] = [255, 0, 0]  # Red boundaries
    Image.fromarray(overlay).save("boundary_overlay.png")

    print("Saved boundary_mask.png and boundary_overlay.png")


if __name__ == "__main__":
    main()
