"""
Test script for patch extraction.
"""

from pathlib import Path
from PIL import Image
import numpy as np
from solution import VillageDataLoader, PatchExtractor


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

    # Initialize patch extractor
    patch_extractor = PatchExtractor(
        imagery_path=village_data.imagery_path,
        boundaries_path=village_data.boundaries_path
    )

    # Extract patch
    print("Extracting patch...")
    patch = patch_extractor.extract(
        plot_geom=plot_geom,
        plot_crs="EPSG:4326",
        buffer_meters=30.0
    )

    print(f"RGB image shape: {patch.rgb_image.shape}")
    print(f"RGB image dtype: {patch.rgb_image.dtype}")
    if patch.boundary_mask is not None:
        print(f"Boundary mask shape: {patch.boundary_mask.shape}")
        print(f"Boundary mask dtype: {patch.boundary_mask.dtype}")
    print(f"Patch CRS: {patch.crs}")
    print(f"Patch bounds: {patch.bounds}")

    # Save RGB image
    print("\nSaving RGB image to patch_rgb.png...")
    Image.fromarray(patch.rgb_image).save("patch_rgb.png")

    # Save boundary mask if available
    if patch.boundary_mask is not None:
        print("Saving boundary mask to patch_boundary.png...")
        # Normalize boundary mask for display
        mask_normalized = ((patch.boundary_mask - patch.boundary_mask.min()) /
                          (patch.boundary_mask.max() - patch.boundary_mask.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(mask_normalized).save("patch_boundary.png")

    print("\nPatch extraction test complete!")


if __name__ == "__main__":
    main()
