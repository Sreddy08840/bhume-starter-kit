"""
Production-ready loader module for village cadastral data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points


def _utm_crs_for(lon: float, lat: float) -> str:
    """Calculate appropriate UTM CRS for a given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    else:
        return f"EPSG:{32700 + zone}"


@dataclass
class VillageData:
    """Container for loaded village data with utility methods."""

    plots: gpd.GeoDataFrame
    imagery_path: Path
    boundaries_path: Optional[Path]
    slug: str
    dir_path: Path

    def get_plot(self, plot_number: str) -> BaseGeometry:
        """
        Get a plot's geometry by plot number.

        Args:
            plot_number: Unique identifier for the plot.

        Returns:
            Plot geometry in EPSG:4326.

        Raises:
            KeyError: If plot number not found.
        """
        plot_number = str(plot_number)
        if plot_number not in self.plots.index:
            raise KeyError(f"Plot number {plot_number} not found")
        return self.plots.loc[plot_number, "geometry"]

    def calculate_area(self, plot_number: str, utm: bool = True) -> float:
        """
        Calculate area of a plot.

        Args:
            plot_number: Unique identifier for the plot.
            utm: If True, calculate area in square meters using UTM projection.
                 If False, calculate in decimal degrees (not recommended for area).

        Returns:
            Area in appropriate units.
        """
        geom = self.get_plot(plot_number)
        if utm:
            utm_crs = _utm_crs_for(geom.centroid.x, geom.centroid.y)
            plots_utm = self.plots.to_crs(utm_crs)
            return float(plots_utm.loc[str(plot_number), "geometry"].area)
        return float(geom.area)

    def find_neighbors(
        self, plot_number: str, distance_m: float = 1.0, max_neighbors: Optional[int] = None
    ) -> list[str]:
        """
        Find neighboring plots for a given plot.

        Args:
            plot_number: Unique identifier for the plot.
            distance_m: Buffer distance in meters to check for neighbors.
            max_neighbors: Maximum number of neighbors to return (closest first).

        Returns:
            List of neighboring plot numbers.
        """
        plot_number = str(plot_number)
        geom = self.get_plot(plot_number)

        # Reproject to UTM for accurate distance calculations
        utm_crs = _utm_crs_for(geom.centroid.x, geom.centroid.y)
        plots_utm = self.plots.to_crs(utm_crs)
        geom_utm = plots_utm.loc[plot_number, "geometry"]

        # Create buffer around plot
        buffer = geom_utm.buffer(distance_m)

        # Find intersecting plots (excluding self)
        mask = (plots_utm.index != plot_number) & (plots_utm.intersects(buffer))
        neighbors = plots_utm[mask].copy()

        # Calculate distance from each neighbor to the plot
        neighbors["distance"] = neighbors.geometry.apply(
            lambda g: geom_utm.distance(g)
        )

        # Sort by distance
        neighbors = neighbors.sort_values("distance")

        # Get plot numbers
        neighbor_numbers = list(neighbors.index)

        if max_neighbors is not None:
            neighbor_numbers = neighbor_numbers[:max_neighbors]

        return neighbor_numbers


class VillageDataLoader:
    """Production-ready loader for village cadastral data bundles."""

    def __init__(self, village_dir: str | Path):
        """
        Initialize the loader with a village directory.

        Args:
            village_dir: Path to the village data directory.
        """
        self.village_dir = Path(village_dir)
        self.slug = self.village_dir.name

    def load(self) -> VillageData:
        """
        Load all village data.

        Returns:
            VillageData object containing loaded data and utility methods.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        # Load input.geojson
        input_path = self.village_dir / "input.geojson"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing required file: {input_path}")

        plots = gpd.read_file(input_path)
        plots["plot_number"] = plots["plot_number"].astype(str)
        plots = plots.set_index("plot_number", drop=False)

        # Ensure plots are in EPSG:4326
        if plots.crs is None:
            plots = plots.set_crs("EPSG:4326")
        elif plots.crs != "EPSG:4326":
            plots = plots.to_crs("EPSG:4326")

        # Check imagery.tif
        imagery_path = self.village_dir / "imagery.tif"
        if not imagery_path.exists():
            raise FileNotFoundError(f"Missing required file: {imagery_path}")

        # Check boundaries.tif (optional)
        boundaries_path = self.village_dir / "boundaries.tif"
        boundaries_path = boundaries_path if boundaries_path.exists() else None

        return VillageData(
            plots=plots,
            imagery_path=imagery_path,
            boundaries_path=boundaries_path,
            slug=self.slug,
            dir_path=self.village_dir,
        )

    def load_imagery(self) -> DatasetReader:
        """
        Load the imagery GeoTIFF as a rasterio dataset.

        Returns:
            Rasterio dataset reader (use as context manager).
        """
        village_data = self.load()
        return rasterio.open(village_data.imagery_path)

    def load_boundaries(self) -> Optional[DatasetReader]:
        """
        Load the boundaries GeoTIFF as a rasterio dataset if available.

        Returns:
            Rasterio dataset reader or None.
        """
        village_data = self.load()
        if village_data.boundaries_path is None:
            return None
        return rasterio.open(village_data.boundaries_path)
