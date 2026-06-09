#!/usr/bin/env python3
"""
Single, self-contained pipeline for cadastral boundary correction.
Combines all modules into one runnable script.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

import numpy as np
import geopandas as gpd
import rasterio
import cv2
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from bhume import load, write_predictions, score


# ==============================================================================
# Helper Functions
# ==============================================================================
def _utm_crs_for(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    else:
        return f"EPSG:{32700 + zone}"


def _reproject_geometry(geom: BaseGeometry, src_crs: str, dst_crs: str) -> BaseGeometry:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class VillageData:
    plots: gpd.GeoDataFrame
    imagery_path: Path
    boundaries_path: Optional[Path]
    slug: str
    dir_path: Path

    def get_plot(self, plot_number: str) -> BaseGeometry:
        plot_number = str(plot_number)
        if plot_number not in self.plots.index:
            raise KeyError(f"Plot number {plot_number} not found")
        return self.plots.loc[plot_number, "geometry"]

    def calculate_area(self, plot_number: str, utm: bool = True) -> float:
        geom = self.get_plot(plot_number)
        if utm:
            utm_crs = _utm_crs_for(geom.centroid.x, geom.centroid.y)
            plots_utm = self.plots.to_crs(utm_crs)
            return float(plots_utm.loc[str(plot_number), "geometry"].area)
        return float(geom.area)

    def find_neighbors(
        self,
        plot_number: str,
        distance_m: float = 1.0,
        max_neighbors: Optional[int] = None,
    ) -> List[str]:
        plot_number = str(plot_number)
        geom = self.get_plot(plot_number)
        utm_crs = _utm_crs_for(geom.centroid.x, geom.centroid.y)
        plots_utm = self.plots.to_crs(utm_crs)
        geom_utm = plots_utm.loc[plot_number, "geometry"]
        buffer = geom_utm.buffer(distance_m)
        neighbors = plots_utm[
            plots_utm.index != plot_number
        ][plots_utm.intersects(buffer)].copy()
        neighbors["distance"] = neighbors.geometry.apply(
            lambda g: geom_utm.distance(g)
        )
        neighbors = neighbors.sort_values("distance")
        neighbor_numbers = list(neighbors.index)
        if max_neighbors is not None:
            neighbor_numbers = neighbor_numbers[:max_neighbors]
        return neighbor_numbers


@dataclass
class Patch:
    rgb_image: np.ndarray
    boundary_mask: Optional[np.ndarray]
    transform: object
    crs: str
    bounds: tuple[float, float, float, float]


@dataclass
class BoundaryDetectionResult:
    binary_mask: np.ndarray
    edge_confidence: float
    contours: List[np.ndarray]


@dataclass
class AlignmentResult:
    best_polygon: BaseGeometry
    dx: float
    dy: float
    rotation: float
    score: float


class AreaFlagReason(Enum):
    OK = "OK"
    AREA_TOO_SMALL = "Area too small"
    AREA_TOO_LARGE = "Area too large"
    INVALID_AREA = "Invalid area (zero or negative)"


@dataclass
class AreaValidationResult:
    area_ratio: float
    area_score: float
    flag_reason: AreaFlagReason
    is_suspicious: bool


@dataclass
class PlotShift:
    plot_number: str
    dx: float
    dy: float


@dataclass
class NeighborConsistencyResult:
    neighbor_score: float
    neighbor_shifts: List[PlotShift]
    direction_similarity: float
    magnitude_similarity: float


@dataclass
class ConfidenceInputs:
    alignment_score: float
    area_score: float
    edge_score: float
    neighbor_score: float


@dataclass
class ConfidenceResult:
    confidence: float
    component_scores: ConfidenceInputs


@dataclass
class PlotPrediction:
    plot_number: str
    status: str
    confidence: float
    method_note: str
    geometry: BaseGeometry


# ==============================================================================
# Village Data Loader
# ==============================================================================
def load_village(village_dir: str | Path) -> VillageData:
    village_dir = Path(village_dir)
    input_path = village_dir / "input.geojson"
    imagery_path = village_dir / "imagery.tif"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing required file: {input_path}")
    if not imagery_path.exists():
        raise FileNotFoundError(f"Missing required file: {imagery_path}")

    plots = gpd.read_file(input_path)
    plots["plot_number"] = plots["plot_number"].astype(str)
    plots = plots.set_index("plot_number", drop=False)

    if plots.crs is None:
        plots = plots.set_crs("EPSG:4326")
    elif plots.crs != "EPSG:4326":
        plots = plots.to_crs("EPSG:4326")

    boundaries_path = village_dir / "boundaries.tif"
    boundaries_path = boundaries_path if boundaries_path.exists() else None

    return VillageData(
        plots=plots,
        imagery_path=imagery_path,
        boundaries_path=boundaries_path,
        slug=village_dir.name,
        dir_path=village_dir,
    )


# ==============================================================================
# Patch Extraction
# ==============================================================================
def extract_patch(
    plot_geom: BaseGeometry,
    imagery_path: Path,
    boundaries_path: Optional[Path] = None,
    buffer_meters: float = 30.0,
    plot_crs: str = "EPSG:4326",
) -> Patch:
    with rasterio.open(imagery_path) as src:
        imagery_crs = str(src.crs)
        geom_imagery = _reproject_geometry(plot_geom, plot_crs, imagery_crs)
        minx, miny, maxx, maxy = geom_imagery.bounds
        left = minx - buffer_meters
        bottom = miny - buffer_meters
        right = maxx + buffer_meters
        top = maxy + buffer_meters
        img_left, img_bottom, img_right, img_top = src.bounds
        left = max(left, img_left)
        bottom = max(bottom, img_bottom)
        right = min(right, img_right)
        top = min(top, img_top)
        if right <= left or top <= bottom:
            raise ValueError("Plot is outside imagery bounds or buffer is too large")
        window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
        rgb = src.read([1, 2, 3], window=window)
        rgb_image = np.transpose(rgb, (1, 2, 0))
        patch_transform = src.window_transform(window)
        patch_bounds = (left, bottom, right, top)

    boundary_mask = None
    if boundaries_path is not None:
        with rasterio.open(boundaries_path) as bsrc:
            if str(bsrc.crs) != imagery_crs:
                raise ValueError("Boundaries raster CRS does not match imagery CRS")
            b_window = rasterio.windows.from_bounds(left, bottom, right, top, transform=bsrc.transform)
            boundary_mask = bsrc.read(1, window=b_window)

    return Patch(
        rgb_image=rgb_image,
        boundary_mask=boundary_mask,
        transform=patch_transform,
        crs=imagery_crs,
        bounds=patch_bounds,
    )


# ==============================================================================
# Boundary Detection
# ==============================================================================
def detect_boundaries(
    rgb_image: np.ndarray,
    boundary_mask: Optional[np.ndarray] = None,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel_size: int = 5,
    morph_kernel_size: int = 3,
) -> BoundaryDetectionResult:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    if boundary_mask is not None:
        hint_normalized = cv2.normalize(
            boundary_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        edges = cv2.addWeighted(edges, 0.7, hint_normalized, 0.3, 0)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    thinned = closed

    contours, _ = cv2.findContours(
        thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    binary_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(binary_mask, contours, -1, 1, 1)

    edge_density = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1])
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_pixels = binary_mask > 0
    if np.sum(edge_pixels) == 0:
        avg_grad = 0.0
    else:
        avg_grad = np.mean(grad_mag[edge_pixels])
    normalized_grad = min(avg_grad / 255.0, 1.0)
    density_score = 1.0 - 2.0 * abs(edge_density - 0.05)
    density_score = max(0.0, min(1.0, density_score))
    edge_confidence = 0.6 * normalized_grad + 0.4 * density_score

    return BoundaryDetectionResult(
        binary_mask=binary_mask,
        edge_confidence=float(edge_confidence),
        contours=contours,
    )


# ==============================================================================
# Plot Alignment
# ==============================================================================
def align_plot(
    official_polygon: BaseGeometry,
    boundary_mask: np.ndarray,
    patch_transform,
    patch_crs: str,
    x_range: tuple[float, float] = (-20.0, 20.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
    rotation_range: tuple[float, float] = (-5.0, 5.0),
    x_step: float = 2.0,
    y_step: float = 2.0,
    rotation_step: float = 1.0,
    polygon_crs: str = "EPSG:4326",
) -> AlignmentResult:
    centroid = official_polygon.centroid
    utm_crs = _utm_crs_for(centroid.x, centroid.y)
    polygon_utm = _reproject_geometry(official_polygon, polygon_crs, utm_crs)
    centroid_utm = polygon_utm.centroid

    best_score = -1.0
    best_dx = 0.0
    best_dy = 0.0
    best_rotation = 0.0
    best_polygon_utm = polygon_utm

    x_offsets = np.arange(x_range[0], x_range[1] + 1e-8, x_step)
    y_offsets = np.arange(y_range[0], y_range[1] + 1e-8, y_step)
    rotations = np.arange(rotation_range[0], rotation_range[1] + 1e-8, rotation_step)

    for dx in x_offsets:
        for dy in y_offsets:
            for rot in rotations:
                transformed = translate(polygon_utm, xoff=dx, yoff=dy)
                transformed = rotate(transformed, rot, origin=centroid_utm)

                try:
                    polygon_patch = _reproject_geometry(transformed, utm_crs, patch_crs)
                    h, w = boundary_mask.shape
                    polygon_mask = np.zeros((h, w), dtype=np.uint8)
                    coords = np.array(polygon_patch.boundary.coords)
                    inv_transform = ~patch_transform
                    pixel_coords = []
                    for x, y in coords:
                        col, row = inv_transform * (x, y)
                        pixel_coords.append((int(round(col)), int(round(row))))

                    if len(pixel_coords) < 2:
                        continue

                    cv2.polylines(
                        polygon_mask,
                        [np.array(pixel_coords, dtype=np.int32)],
                        isClosed=True,
                        color=1,
                        thickness=2,
                    )
                    overlap = np.sum((polygon_mask > 0) & (boundary_mask > 0))
                    total_boundary = np.sum(polygon_mask > 0)
                    if total_boundary == 0:
                        continue

                    score = float(overlap / total_boundary)
                    if score > best_score:
                        best_score = score
                        best_dx = dx
                        best_dy = dy
                        best_rotation = rot
                        best_polygon_utm = transformed
                except Exception:
                    continue

    best_polygon = _reproject_geometry(best_polygon_utm, utm_crs, polygon_crs)

    return AlignmentResult(
        best_polygon=best_polygon,
        dx=best_dx,
        dy=best_dy,
        rotation=best_rotation,
        score=best_score,
    )


# ==============================================================================
# Area Validation
# ==============================================================================
def validate_area(
    recorded_area: float,
    polygon_area: float,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> AreaValidationResult:
    if recorded_area <= 0 or polygon_area <= 0:
        return AreaValidationResult(
            area_ratio=0.0,
            area_score=0.0,
            flag_reason=AreaFlagReason.INVALID_AREA,
            is_suspicious=True,
        )

    area_ratio = polygon_area / recorded_area

    distance = abs(1.0 - area_ratio)
    k = 3.0
    area_score = float(max(0.0, min(1.0, np.exp(-k * distance))))

    flag_reason = AreaFlagReason.OK
    is_suspicious = False

    if area_ratio < min_ratio:
        flag_reason = AreaFlagReason.AREA_TOO_SMALL
        is_suspicious = True
    elif area_ratio > max_ratio:
        flag_reason = AreaFlagReason.AREA_TOO_LARGE
        is_suspicious = True

    return AreaValidationResult(
        area_ratio=area_ratio,
        area_score=area_score,
        flag_reason=flag_reason,
        is_suspicious=is_suspicious,
    )


# ==============================================================================
# Neighbor Consistency
# ==============================================================================
def compute_neighbor_score(
    plot_shift: PlotShift,
    neighbor_shifts: List[PlotShift],
) -> NeighborConsistencyResult:
    if not neighbor_shifts:
        return NeighborConsistencyResult(
            neighbor_score=0.5,
            neighbor_shifts=[],
            direction_similarity=0.5,
            magnitude_similarity=0.5,
        )

    direction_similarities = []
    plot_vec = np.array([plot_shift.dx, plot_shift.dy])
    plot_mag = np.linalg.norm(plot_vec)

    for neighbor in neighbor_shifts:
        neighbor_vec = np.array([neighbor.dx, neighbor.dy])
        neighbor_mag = np.linalg.norm(neighbor_vec)

        if plot_mag < 1e-8 or neighbor_mag < 1e-8:
            dir_sim = 1.0
        else:
            dir_sim = float(np.dot(plot_vec, neighbor_vec) / (plot_mag * neighbor_mag))
            dir_sim = max(-1.0, min(1.0, dir_sim))
        direction_similarities.append(dir_sim)

    magnitude_similarities = []
    for neighbor in neighbor_shifts:
        neighbor_vec = np.array([neighbor.dx, neighbor.dy])
        neighbor_mag = np.linalg.norm(neighbor_vec)
        max_mag = max(plot_mag, neighbor_mag, 1e-8)
        mag_diff = abs(plot_mag - neighbor_mag)
        mag_sim = 1.0 - (mag_diff / max_mag)
        mag_sim = max(0.0, min(1.0, mag_sim))
        magnitude_similarities.append(mag_sim)

    avg_dir_sim = float(np.mean(direction_similarities))
    avg_mag_sim = float(np.mean(magnitude_similarities))
    normalized_dir_sim = (avg_dir_sim + 1.0) / 2.0
    neighbor_score = 0.6 * normalized_dir_sim + 0.4 * avg_mag_sim

    return NeighborConsistencyResult(
        neighbor_score=neighbor_score,
        neighbor_shifts=neighbor_shifts,
        direction_similarity=avg_dir_sim,
        magnitude_similarity=avg_mag_sim,
    )


# ==============================================================================
# Confidence Calibration
# ==============================================================================
def compute_confidence(
    inputs: ConfidenceInputs,
) -> ConfidenceResult:
    alignment_score = np.clip(inputs.alignment_score, 0.0, 1.0)
    area_score = np.clip(inputs.area_score, 0.0, 1.0)
    edge_score = np.clip(inputs.edge_score, 0.0, 1.0)
    neighbor_score = np.clip(inputs.neighbor_score, 0.0, 1.0)

    weighted_score = (
        0.35 * alignment_score
        + 0.25 * area_score
        + 0.25 * edge_score
        + 0.15 * neighbor_score
    )

    min_score = min(alignment_score, area_score, edge_score, neighbor_score)
    confidence = float(weighted_score * (0.7 + 0.3 * min_score))
    confidence = np.clip(confidence, 0.0, 1.0)

    return ConfidenceResult(
        confidence=confidence,
        component_scores=ConfidenceInputs(
            alignment_score=alignment_score,
            area_score=area_score,
            edge_score=edge_score,
            neighbor_score=neighbor_score,
        ),
    )


# ==============================================================================
# Prediction Generation
# ==============================================================================
def create_prediction(
    plot_number: str,
    confidence: float,
    corrected_geometry: BaseGeometry,
    original_geometry: BaseGeometry,
    method_note: str = "",
    corrected_threshold: float = 0.7,
) -> PlotPrediction:
    confidence = max(0.0, min(1.0, confidence))
    if confidence > corrected_threshold:
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


def generate_predictions_gdf(
    plot_predictions: List[PlotPrediction],
) -> gpd.GeoDataFrame:
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


# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: uv run single_pipeline.py <village_directory> [output_path]")
        print("Example: uv run single_pipeline.py data/34855_vadnerbhairav_chandavad_nashik")
        sys.exit(1)

    village_dir = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else village_dir / "predictions.geojson"

    print("=" * 80)
    print("Starting Cadastral Boundary Correction Pipeline")
    print("=" * 80)

    # Step 1: Load village
    print("\n[1/10] Loading village data...")
    village_data = load_village(village_dir)
    print(f"   Loaded {len(village_data.plots)} plots")

    # Initialize storage
    plot_shifts: Dict[str, PlotShift] = {}
    original_geometries: Dict[str, BaseGeometry] = {}
    corrected_geometries: Dict[str, BaseGeometry] = {}
    area_scores: Dict[str, float] = {}
    alignment_scores: Dict[str, float] = {}
    edge_scores: Dict[str, float] = {}

    # Step 2-6: Iterate plots, extract patches, detect boundaries, align, compute area score
    print("\n[2/10] Processing plots (extracting patches, detecting boundaries, aligning)...")
    for i, plot_num in enumerate(village_data.plots.index):
        if i % 10 == 0:
            print(f"   Processed {i}/{len(village_data.plots)} plots...")

        plot_geom = village_data.get_plot(plot_num)
        original_geometries[plot_num] = plot_geom

        try:
            # Step 3: Extract patch
            patch = extract_patch(
                plot_geom,
                village_data.imagery_path,
                village_data.boundaries_path,
                buffer_meters=30.0,
            )

            # Step 4: Detect boundaries
            detection_result = detect_boundaries(patch.rgb_image, patch.boundary_mask)
            edge_scores[plot_num] = detection_result.edge_confidence

            # Step 5: Align plot
            alignment_result = align_plot(
                plot_geom,
                detection_result.binary_mask,
                patch.transform,
                patch.crs,
            )
            alignment_scores[plot_num] = alignment_result.score
            corrected_geometries[plot_num] = alignment_result.best_polygon
            plot_shifts[plot_num] = PlotShift(
                plot_number=plot_num,
                dx=alignment_result.dx,
                dy=alignment_result.dy,
            )

            # Step 6: Compute area score
            recorded_area = village_data.calculate_area(plot_num)
            utm_crs = _utm_crs_for(plot_geom.centroid.x, plot_geom.centroid.y)
            aligned_geom_utm = _reproject_geometry(alignment_result.best_polygon, "EPSG:4326", utm_crs)
            aligned_area = aligned_geom_utm.area
            area_result = validate_area(recorded_area, aligned_area)
            area_scores[plot_num] = area_result.area_score
        except Exception as e:
            print(f"   Warning: Failed to process plot {plot_num}: {e}")
            corrected_geometries[plot_num] = plot_geom
            plot_shifts[plot_num] = PlotShift(plot_number=plot_num, dx=0.0, dy=0.0)
            area_scores[plot_num] = 0.0
            alignment_scores[plot_num] = 0.0
            edge_scores[plot_num] = 0.0

    print(f"   Processed {len(village_data.plots)}/{len(village_data.plots)} plots.")

    # Step 7-9: Compute neighbor score, confidence, correct or flag
    print("\n[7-9/10] Computing neighbor scores, confidence, and generating predictions...")
    plot_predictions: List[PlotPrediction] = []

    for plot_num in village_data.plots.index:
        plot_geom = original_geometries[plot_num]
        neighbors = village_data.find_neighbors(plot_num, distance_m=1.0)
        neighbor_shifts = [plot_shifts[n] for n in neighbors if n in plot_shifts]

        plot_shift = plot_shifts[plot_num]
        neighbor_result = compute_neighbor_score(plot_shift, neighbor_shifts)

        confidence_inputs = ConfidenceInputs(
            alignment_score=alignment_scores.get(plot_num, 0.0),
            area_score=area_scores.get(plot_num, 0.0),
            edge_score=edge_scores.get(plot_num, 0.0),
            neighbor_score=neighbor_result.neighbor_score,
        )
        confidence_result = compute_confidence(confidence_inputs)

        prediction = create_prediction(
            plot_number=plot_num,
            confidence=confidence_result.confidence,
            corrected_geometry=corrected_geometries[plot_num],
            original_geometry=plot_geom,
            method_note=f"dx={plot_shift.dx:.2f}m, dy={plot_shift.dy:.2f}m",
            corrected_threshold=0.7,
        )
        plot_predictions.append(prediction)

    # Step 10: Write predictions.geojson
    print("\n[10/10] Generating predictions GeoDataFrame and saving...")
    predictions_gdf = generate_predictions_gdf(plot_predictions)
    write_predictions(output_path, predictions_gdf)

    # Print summary
    print("\n" + "=" * 80)
    print("Pipeline Complete! Summary:")
    print("=" * 80)
    print(f"   Total plots: {len(predictions_gdf)}")
    corrected_count = len(predictions_gdf[predictions_gdf["status"] == "corrected"])
    flagged_count = len(predictions_gdf[predictions_gdf["status"] == "flagged"])
    print(f"   Corrected: {corrected_count}")
    print(f"   Flagged: {flagged_count}")
    print(f"\n   Predictions saved to: {output_path}")

    # Try to score against example truths
    try:
        bhume_village = load(village_dir)
        if bhume_village.example_truths is not None:
            print("\nScoring predictions against example truths...")
            score_result = score(predictions_gdf, bhume_village)
            print("\n" + str(score_result))
    except Exception as e:
        print(f"\nNote: Could not score predictions: {e}")


if __name__ == "__main__":
    main()
