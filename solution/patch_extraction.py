"""
Patch extraction module for cadastral boundary correction.
Extracts image and boundary patches around plot geometries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform


@dataclass
class Patch:
    """
    Dataclass containing extracted patch data.

    Attributes:
        rgb_image: RGB image as (H, W, 3) uint8 numpy array
        boundary_mask: Boundary mask as (H, W) numpy array (or None)
        transform: Affine transform mapping pixel coordinates to imagery CRS
        crs: CRS of the patch (as string)
        bounds: Bounding box of patch in imagery CRS (left, bottom, right, top)
    """
    rgb_image: np.ndarray
    boundary_mask: Optional[np.ndarray]
    transform: Affine
    crs: str
    bounds: Tuple[float, float, float, float]


class PatchExtractor:
    """
    Extracts image and boundary patches around plot geometries.
    """

    def __init__(
        self,
        imagery_path: str | Path,
        boundaries_path: Optional[str | Path] = None,
    ):
        """
        Initialize the patch extractor.

        Args:
            imagery_path: Path to imagery.tif
            boundaries_path: Path to boundaries.tif (optional)
        """
        self.imagery_path = Path(imagery_path)
        self.boundaries_path = Path(boundaries_path) if boundaries_path is not None else None

    def _reproject_geometry(
        self,
        geom: BaseGeometry,
        src_crs: str,
        dst_crs: str,
    ) -> BaseGeometry:
        """
        Reproject a geometry from source CRS to destination CRS.

        Args:
            geom: Input geometry
            src_crs: Source CRS (e.g., "EPSG:4326")
            dst_crs: Destination CRS (e.g., "EPSG:3857")

        Returns:
            Reprojected geometry
        """
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)

    def extract(
        self,
        plot_geom: BaseGeometry,
        plot_crs: str = "EPSG:4326",
        buffer_meters: float = 25.0,
    ) -> Patch:
        """
        Extract patch around a plot geometry.

        Args:
            plot_geom: Plot geometry (polygon)
            plot_crs: CRS of the plot geometry (default: EPSG:4326)
            buffer_meters: Buffer distance in meters to add around plot

        Returns:
            Patch object containing RGB image, boundary mask, transform, CRS, and bounds

        Raises:
            ValueError: If plot is outside imagery bounds
        """
        # Open imagery dataset
        with rasterio.open(self.imagery_path) as src:
            imagery_crs = str(src.crs)

            # Reproject plot to imagery CRS
            geom_imagery = self._reproject_geometry(plot_geom, plot_crs, imagery_crs)

            # Get plot bounds and add buffer
            minx, miny, maxx, maxy = geom_imagery.bounds
            left = minx - buffer_meters
            bottom = miny - buffer_meters
            right = maxx + buffer_meters
            top = maxy + buffer_meters

            # Clip patch bounds to imagery bounds
            img_left, img_bottom, img_right, img_top = src.bounds
            left = max(left, img_left)
            bottom = max(bottom, img_bottom)
            right = min(right, img_right)
            top = min(top, img_top)

            # Check if patch is valid
            if right <= left or top <= bottom:
                raise ValueError("Plot is outside imagery bounds or buffer is too large")

            # Read RGB patch
            window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
            rgb = src.read([1, 2, 3], window=window)
            # Convert from (3, H, W) to (H, W, 3)
            rgb_image = np.transpose(rgb, (1, 2, 0))

            # Get patch transform and bounds
            patch_transform = src.window_transform(window)
            patch_bounds = (left, bottom, right, top)

        # Extract boundary mask if available
        boundary_mask = None
        if self.boundaries_path is not None:
            with rasterio.open(self.boundaries_path) as bsrc:
                # Ensure boundary raster matches imagery CRS
                if str(bsrc.crs) != imagery_crs:
                    raise ValueError("Boundaries raster CRS does not match imagery CRS")

                # Read boundary patch
                b_window = rasterio.windows.from_bounds(left, bottom, right, top, transform=bsrc.transform)
                boundary_mask = bsrc.read(1, window=b_window)

        return Patch(
            rgb_image=rgb_image,
            boundary_mask=boundary_mask,
            transform=patch_transform,
            crs=imagery_crs,
            bounds=patch_bounds,
        )
