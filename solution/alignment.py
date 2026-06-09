"""
Plot alignment module: finds best translation and rotation to align plot with boundary mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer


def _utm_crs_for(lon: float, lat: float) -> str:
    """Calculate appropriate UTM CRS for a given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    else:
        return f"EPSG:{32700 + zone}"


def _reproject_geometry(geom: BaseGeometry, src_crs: str, dst_crs: str) -> BaseGeometry:
    """Reproject a geometry from source CRS to destination CRS."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    from shapely.ops import transform as shp_transform
    return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


@dataclass
class AlignmentResult:
    """
    Result of plot alignment.

    Attributes:
        best_polygon: Aligned polygon (in original CRS)
        dx: Translation in x-direction (meters)
        dy: Translation in y-direction (meters)
        rotation: Rotation in degrees
        score: Alignment quality score (higher is better)
    """
    best_polygon: BaseGeometry
    dx: float
    dy: float
    rotation: float
    score: float


class PlotAligner:
    """
    Aligns official plot polygons with detected boundary masks.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-20.0, 20.0),
        y_range: tuple[float, float] = (-20.0, 20.0),
        rotation_range: tuple[float, float] = (-5.0, 5.0),
        x_step: float = 2.0,
        y_step: float = 2.0,
        rotation_step: float = 1.0,
    ):
        """
        Initialize the plot aligner.

        Args:
            x_range: Min and max x offset (meters)
            y_range: Min and max y offset (meters)
            rotation_range: Min and max rotation (degrees)
            x_step: Step size for x offset search
            y_step: Step size for y offset search
            rotation_step: Step size for rotation search
        """
        self.x_range = x_range
        self.y_range = y_range
        self.rotation_range = rotation_range
        self.x_step = x_step
        self.y_step = y_step
        self.rotation_step = rotation_step

    def align(
        self,
        official_polygon: BaseGeometry,
        boundary_mask: np.ndarray,
        patch_transform: 'Affine',
        patch_crs: str,
        polygon_crs: str = "EPSG:4326",
    ) -> AlignmentResult:
        """
        Align official plot polygon with boundary mask.

        Args:
            official_polygon: Official plot geometry (polygon)
            boundary_mask: Binary boundary mask (H, W)
            patch_transform: Affine transform mapping pixel coordinates to patch CRS
            patch_crs: CRS of the patch
            polygon_crs: CRS of the official polygon

        Returns:
            AlignmentResult containing best polygon, dx, dy, rotation, and score
        """
        # Get polygon centroid for UTM zone
        centroid = official_polygon.centroid
        utm_crs = _utm_crs_for(centroid.x, centroid.y)

        # Reproject polygon to UTM (meters) for accurate transformations
        polygon_utm = _reproject_geometry(official_polygon, polygon_crs, utm_crs)
        centroid_utm = polygon_utm.centroid

        best_score = -1.0
        best_dx = 0.0
        best_dy = 0.0
        best_rotation = 0.0
        best_polygon_utm = polygon_utm

        # Generate search ranges
        x_offsets = np.arange(self.x_range[0], self.x_range[1] + 1e-8, self.x_step)
        y_offsets = np.arange(self.y_range[0], self.y_range[1] + 1e-8, self.y_step)
        rotations = np.arange(self.rotation_range[0], self.rotation_range[1] + 1e-8, self.rotation_step)

        # Search grid
        for dx in x_offsets:
            for dy in y_offsets:
                for rot in rotations:
                    # Transform polygon
                    transformed = translate(polygon_utm, xoff=dx, yoff=dy)
                    transformed = rotate(transformed, rot, origin=centroid_utm)

                    # Calculate score
                    score = self._calculate_score(
                        transformed,
                        boundary_mask,
                        patch_transform,
                        patch_crs,
                        utm_crs,
                    )

                    # Update best
                    if score > best_score:
                        best_score = score
                        best_dx = dx
                        best_dy = dy
                        best_rotation = rot
                        best_polygon_utm = transformed

        # Reproject best polygon back to original CRS
        best_polygon = _reproject_geometry(best_polygon_utm, utm_crs, polygon_crs)

        return AlignmentResult(
            best_polygon=best_polygon,
            dx=best_dx,
            dy=best_dy,
            rotation=best_rotation,
            score=best_score,
        )

    def _calculate_score(
        self,
        polygon_utm: BaseGeometry,
        boundary_mask: np.ndarray,
        patch_transform: 'Affine',
        patch_crs: str,
        utm_crs: str,
    ) -> float:
        """
        Calculate alignment score by checking how much polygon boundary overlaps with mask.

        Args:
            polygon_utm: Plot polygon in UTM CRS
            boundary_mask: Binary boundary mask
            patch_transform: Affine transform for patch
            patch_crs: CRS of patch
            utm_crs: UTM CRS of polygon

        Returns:
            Alignment score (higher is better)
        """
        try:
            # Reproject polygon to patch CRS
            polygon_patch = _reproject_geometry(polygon_utm, utm_crs, patch_crs)

            # Create a raster mask of the polygon boundary
            h, w = boundary_mask.shape
            polygon_mask = np.zeros((h, w), dtype=np.uint8)

            # Get polygon boundary coordinates in pixel space
            coords = np.array(polygon_patch.boundary.coords)

            # Convert coordinates to pixels
            inv_transform = ~patch_transform
            pixel_coords = []
            for x, y in coords:
                col, row = inv_transform * (x, y)
                pixel_coords.append((int(round(col)), int(round(row))))

            if len(pixel_coords) < 2:
                return 0.0

            # Draw polygon boundary on mask
            import cv2
            cv2.polylines(
                polygon_mask,
                [np.array(pixel_coords, dtype=np.int32)],
                isClosed=True,
                color=1,
                thickness=2,
            )

            # Calculate overlap
            overlap = np.sum((polygon_mask > 0) & (boundary_mask > 0))
            total_boundary = np.sum(polygon_mask > 0)

            if total_boundary == 0:
                return 0.0

            # Return overlap ratio
            return float(overlap / total_boundary)

        except Exception:
            return 0.0
