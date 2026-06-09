"""
Test script for plot alignment.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from solution import (
    VillageDataLoader,
    PatchExtractor,
    FieldBoundaryDetector,
    PlotAligner,
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
    detection_result = detector.detect(patch.rgb_image, patch.boundary_mask)
    print(f"Edge confidence: {detection_result.edge_confidence:.4f}")

    # Align plot
    print("\nAligning plot...")
    aligner = PlotAligner()
    alignment_result = aligner.align(
        official_polygon=plot_geom,
        boundary_mask=detection_result.binary_mask,
        patch_transform=patch.transform,
        patch_crs=patch.crs,
    )

    print("\nAlignment complete!")
    print(f"Best dx: {alignment_result.dx:.2f} m")
    print(f"Best dy: {alignment_result.dy:.2f} m")
    print(f"Best rotation: {alignment_result.rotation:.2f} degrees")
    print(f"Alignment score: {alignment_result.score:.4f}")

    # Save visualization
    print("\nSaving visualization...")
    img = Image.fromarray(patch.rgb_image.copy())
    draw = ImageDraw.Draw(img)

    # Draw original plot boundary on image
    def draw_polygon(polygon, color, width=3):
        # Reproject to patch CRS
        from solution.alignment import _reproject_geometry
        poly_patch = _reproject_geometry(polygon, "EPSG:4326", patch.crs)
        coords = list(poly_patch.boundary.coords)
        inv_transform = ~patch.transform
        pixel_coords = []
        for x, y in coords:
            col, row = inv_transform * (x, y)
            pixel_coords.append((col, row))
        draw.line(pixel_coords, fill=color, width=width)

    draw_polygon(plot_geom, color=(0, 255, 0), width=3)  # Green - original
    draw_polygon(alignment_result.best_polygon, color=(255, 0, 0), width=3)  # Red - aligned
    img.save("alignment_result.png")

    print("Saved alignment_result.png")


if __name__ == "__main__":
    main()
