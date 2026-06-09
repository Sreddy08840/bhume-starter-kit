#!/usr/bin/env python3
"""
Upgraded, Gold/Platinum level pipeline for cadastral boundary correction.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

import numpy as np
import geopandas as gpd
import rasterio
import cv2
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import transform as shp_transform, nearest_points
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

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
    example_truths: Optional[gpd.GeoDataFrame]
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
        # Combine masks into one to avoid reindexing warning
        mask = (plots_utm.index != plot_number) & (plots_utm.intersects(buffer))
        neighbors = plots_utm[mask].copy()
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
    edge_strength_map: np.ndarray


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


@dataclass
class GlobalAlignmentResult:
    dx: float
    dy: float
    rotation: float
    method: str


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

    truths_path = village_dir / "example_truths.geojson"
    example_truths = None
    if truths_path.exists():
        example_truths = gpd.read_file(truths_path)
        example_truths["plot_number"] = example_truths["plot_number"].astype(str)
        example_truths = example_truths.set_index("plot_number", drop=False)

    return VillageData(
        plots=plots,
        imagery_path=imagery_path,
        boundaries_path=boundaries_path,
        example_truths=example_truths,
        slug=village_dir.name,
        dir_path=village_dir,
    )


# ==============================================================================
# Global Alignment
# ==============================================================================
def compute_global_alignment(village_data: VillageData) -> GlobalAlignmentResult:
    if village_data.example_truths is not None:
        return _compute_global_alignment_from_truths(village_data)
    else:
        return GlobalAlignmentResult(dx=0.0, dy=0.0, rotation=0.0, method="no_truths")


def _compute_global_alignment_from_truths(village_data: VillageData) -> GlobalAlignmentResult:
    truths = village_data.example_truths
    plots = village_data.plots
    common_plots = [p for p in truths.index if p in plots.index]
    if not common_plots:
        return GlobalAlignmentResult(dx=0.0, dy=0.0, rotation=0.0, method="no_common_plots")

    utm_crs = _utm_crs_for(plots.loc[common_plots[0], "geometry"].centroid.x, plots.loc[common_plots[0], "geometry"].centroid.y)
    plots_utm = plots.to_crs(utm_crs)
    truths_utm = truths.to_crs(utm_crs)

    dxs, dys = [], []
    for plot_num in common_plots:
        p_centroid = plots_utm.loc[plot_num, "geometry"].centroid
        t_centroid = truths_utm.loc[plot_num, "geometry"].centroid
        dxs.append(t_centroid.x - p_centroid.x)
        dys.append(t_centroid.y - p_centroid.y)

    if not dxs or not dys:
        return GlobalAlignmentResult(dx=0.0, dy=0.0, rotation=0.0, method="no_truths")

    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    return GlobalAlignmentResult(dx=dx, dy=dy, rotation=0.0, method="median_shift_from_truths")


# ==============================================================================
# Patch Extraction
# ==============================================================================
def extract_patch(
    plot_geom: BaseGeometry,
    imagery_path: Path,
    boundaries_path: Optional[Path] = None,
    buffer_meters: float = 40.0,
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
        try:
            with rasterio.open(boundaries_path) as bsrc:
                if str(bsrc.crs) != imagery_crs:
                    pass  # Skip if CRS doesn't match
                else:
                    b_window = rasterio.windows.from_bounds(left, bottom, right, top, transform=bsrc.transform)
                    boundary_mask = bsrc.read(1, window=b_window)
        except Exception:
            boundary_mask = None

    return Patch(
        rgb_image=rgb_image,
        boundary_mask=boundary_mask,
        transform=patch_transform,
        crs=imagery_crs,
        bounds=patch_bounds,
    )


# ==============================================================================
# Improved Boundary Detection
# ==============================================================================
def detect_boundaries(
    rgb_image: np.ndarray,
    boundary_mask: Optional[np.ndarray] = None,
    blur_kernel_size: int = 5,
    morph_kernel_size: int = 3,
) -> BoundaryDetectionResult:
    # Convert to LAB and enhance contrast on L channel
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2GRAY)

    # Adaptive thresholding and multi-scale Canny
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Compute adaptive Canny thresholds using median of pixel intensities
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)

    # Multi-scale edges
    edges2 = cv2.Canny(cv2.GaussianBlur(blurred, (9, 9), 0), lower // 2, upper // 2)
    edges_combined = cv2.bitwise_or(edges, edges2)

    # Incorporate boundary hint with adaptive weight
    if boundary_mask is not None:
        try:
            # Resize boundary_mask to match edges_combined size
            target_size = (edges_combined.shape[1], edges_combined.shape[0])
            hint_normalized = cv2.normalize(
                boundary_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            if hint_normalized.shape != target_size:
                hint_normalized = cv2.resize(hint_normalized, target_size, interpolation=cv2.INTER_LINEAR)
            # Compute hint quality (how structured it is)
            hint_edges = cv2.Canny(hint_normalized, 50, 150)
            hint_density = np.sum(hint_edges > 0) / (hint_edges.shape[0] * hint_edges.shape[1])
            hint_weight = min(0.5, max(0.1, hint_density * 10))
            edges_combined = cv2.addWeighted(edges_combined, 1 - hint_weight, hint_normalized, hint_weight, 0)
        except Exception:
            # Skip boundary hint if it fails, just use edges_combined
            pass

    # Morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    try:
        thinned = cv2.ximgproc.thinning(opened) if hasattr(cv2, 'ximgproc') else opened
    except Exception:
        thinned = opened

    contours, _ = cv2.findContours(
        thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    binary_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(binary_mask, contours, -1, 1, 1)

    # Compute edge confidence
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
        edge_strength_map=grad_mag,
    )


# ==============================================================================
# Improved Plot Alignment (with global initial guess + local refinement)
# ==============================================================================
def align_plot(
    official_polygon: BaseGeometry,
    boundary_mask: np.ndarray,
    patch_transform,
    patch_crs: str,
    global_shift: Optional[Tuple[float, float]] = None,
    polygon_crs: str = "EPSG:4326",
) -> Tuple[AlignmentResult, float]:
    centroid = official_polygon.centroid
    utm_crs = _utm_crs_for(centroid.x, centroid.y)
    polygon_utm = _reproject_geometry(official_polygon, polygon_crs, utm_crs)
    centroid_utm = polygon_utm.centroid

    # Start with global shift if available
    if global_shift is not None:
        dx_init, dy_init = global_shift
    else:
        dx_init, dy_init = 0.0, 0.0

    best_score = -1.0
    best_dx = dx_init
    best_dy = dy_init
    best_rotation = 0.0
    best_polygon_utm = polygon_utm

    # Coarse grid search around global shift
    search_range = 15.0
    x_offsets = np.arange(dx_init - search_range, dx_init + search_range + 1e-8, 3.0)
    y_offsets = np.arange(dy_init - search_range, dy_init + search_range + 1e-8, 3.0)
    rotations = np.arange(-3.0, 3.0 + 1e-8, 1.0)

    for dx in x_offsets:
        for dy in y_offsets:
            for rot in rotations:
                transformed = translate(polygon_utm, xoff=dx, yoff=dy)
                transformed = rotate(transformed, rot, origin=centroid_utm)
                score = _compute_alignment_score(transformed, boundary_mask, patch_transform, patch_crs, utm_crs)
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy
                    best_rotation = rot
                    best_polygon_utm = transformed

    # Fine local refinement
    best_score, best_dx, best_dy, best_rotation, best_polygon_utm = _local_refinement(
        polygon_utm,
        centroid_utm,
        boundary_mask,
        patch_transform,
        patch_crs,
        utm_crs,
        best_dx,
        best_dy,
        best_rotation,
        best_score,
    )

    best_polygon = _reproject_geometry(best_polygon_utm, utm_crs, polygon_crs)
    relative_dx = best_dx - (dx_init if global_shift else 0.0)
    relative_dy = best_dy - (dy_init if global_shift else 0.0)
    return AlignmentResult(
        best_polygon=best_polygon,
        dx=best_dx,
        dy=best_dy,
        rotation=best_rotation,
        score=best_score,
    ), best_score


def _compute_alignment_score(
    polygon_utm: BaseGeometry,
    boundary_mask: np.ndarray,
    patch_transform,
    patch_crs: str,
    utm_crs: str,
) -> float:
    try:
        polygon_patch = _reproject_geometry(polygon_utm, utm_crs, patch_crs)
        h, w = boundary_mask.shape
        polygon_mask = np.zeros((h, w), dtype=np.uint8)
        coords = np.array(polygon_patch.boundary.coords)
        inv_transform = ~patch_transform
        pixel_coords = []
        for x, y in coords:
            col, row = inv_transform * (x, y)
            pixel_coords.append((int(round(col)), int(round(row))))
        if len(pixel_coords) < 2:
            return 0.0
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
            return 0.0
        return float(overlap / total_boundary)
    except Exception:
        return 0.0


def _local_refinement(
    polygon_utm,
    centroid_utm,
    boundary_mask,
    patch_transform,
    patch_crs,
    utm_crs,
    best_dx,
    best_dy,
    best_rotation,
    best_score,
):
    # Initialize with the current best guess
    best_polygon_utm = translate(polygon_utm, xoff=best_dx, yoff=best_dy)
    best_polygon_utm = rotate(best_polygon_utm, best_rotation, origin=centroid_utm)
    
    # Refine translation
    step = 1.0
    for _ in range(3):
        improved = False
        for dx_delta in [-step, 0, step]:
            for dy_delta in [-step, 0, step]:
                dx = best_dx + dx_delta
                dy = best_dy + dy_delta
                transformed = translate(polygon_utm, xoff=dx, yoff=dy)
                transformed = rotate(transformed, best_rotation, origin=centroid_utm)
                score = _compute_alignment_score(transformed, boundary_mask, patch_transform, patch_crs, utm_crs)
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy
                    best_polygon_utm = transformed
                    improved = True
        if not improved:
            step /= 2
        else:
            best_polygon_utm = translate(polygon_utm, xoff=best_dx, yoff=best_dy)
            best_polygon_utm = rotate(best_polygon_utm, best_rotation, origin=centroid_utm)

    # Refine rotation
    step = 0.5
    for _ in range(3):
        improved = False
        for rot_delta in [-step, 0, step]:
            rot = best_rotation + rot_delta
            transformed = translate(polygon_utm, xoff=best_dx, yoff=best_dy)
            transformed = rotate(transformed, rot, origin=centroid_utm)
            score = _compute_alignment_score(transformed, boundary_mask, patch_transform, patch_crs, utm_crs)
            if score > best_score:
                best_score = score
                best_rotation = rot
                best_polygon_utm = transformed
                improved = True
        if not improved:
            step /= 2
        else:
            best_polygon_utm = translate(polygon_utm, xoff=best_dx, yoff=best_dy)
            best_polygon_utm = rotate(best_polygon_utm, best_rotation, origin=centroid_utm)

    return best_score, best_dx, best_dy, best_rotation, best_polygon_utm


@dataclass
class AlignmentResult:
    best_polygon: BaseGeometry
    dx: float
    dy: float
    rotation: float
    score: float


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
    log_ratio = np.log(area_ratio)
    area_score = float(np.exp(-3.0 * (log_ratio ** 2)))

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
        if plot_mag < 1e-8 and neighbor_mag < 1e-8:
            dir_sim = 1.0
        elif plot_mag < 1e-8 or neighbor_mag < 1e-8:
            dir_sim = 0.5
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
# Confidence Calibration (with Isotonic Regression if truths available)
# ==============================================================================
class CalibratedConfidence:
    def __init__(self):
        self.isotonic_regressor: Optional[IsotonicRegression] = None
        self._weights = np.array([0.35, 0.25, 0.25, 0.15])

    def fit(self, inputs_list: List[ConfidenceInputs], iou_list: List[float]):
        raw_scores = [self._compute_raw_score(inp) for inp in inputs_list]
        self.isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
        self.isotonic_regressor.fit(raw_scores, iou_list)

    def predict(self, inputs: ConfidenceInputs) -> float:
        raw = self._compute_raw_score(inputs)
        if self.isotonic_regressor is not None:
            cal = self.isotonic_regressor.predict([raw])[0]
            return float(max(0.0, min(1.0, cal)))
        else:
            min_score = min(inputs.alignment_score, inputs.area_score, inputs.edge_score, inputs.neighbor_score)
            return float(max(0.0, min(1.0, raw * (0.7 + 0.3 * min_score))))

    def _compute_raw_score(self, inputs: ConfidenceInputs) -> float:
        scores = np.array([
            np.clip(inputs.alignment_score, 0.0, 1.0),
            np.clip(inputs.area_score, 0.0, 1.0),
            np.clip(inputs.edge_score, 0.0, 1.0),
            np.clip(inputs.neighbor_score, 0.0, 1.0),
        ])
        return float(np.sum(scores * self._weights))


def compute_confidence(
    inputs: ConfidenceInputs,
    calibrator: Optional[CalibratedConfidence] = None,
) -> ConfidenceResult:
    if calibrator is not None:
        conf = calibrator.predict(inputs)
    else:
        min_score = min(inputs.alignment_score, inputs.area_score, inputs.edge_score, inputs.neighbor_score)
        scores = np.array([
            np.clip(inputs.alignment_score, 0.0, 1.0),
            np.clip(inputs.area_score, 0.0, 1.0),
            np.clip(inputs.edge_score, 0.0, 1.0),
            np.clip(inputs.neighbor_score, 0.0, 1.0),
        ])
        weights = np.array([0.35, 0.25, 0.25, 0.15])
        raw = float(np.sum(scores * weights))
        conf = float(max(0.0, min(1.0, raw * (0.7 + 0.3 * min_score))))

    return ConfidenceResult(
        confidence=conf,
        component_scores=ConfidenceInputs(
            alignment_score=np.clip(inputs.alignment_score, 0.0, 1.0),
            area_score=np.clip(inputs.area_score, 0.0, 1.0),
            edge_score=np.clip(inputs.edge_score, 0.0, 1.0),
            neighbor_score=np.clip(inputs.neighbor_score, 0.0, 1.0),
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
        print("Usage: uv run upgraded_pipeline.py <village_directory> [output_path]")
        print("Example: uv run upgraded_pipeline.py data/34855_vadnerbhairav_chandavad_nashik")
        sys.exit(1)

    village_dir = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else village_dir / "predictions.geojson"

    print("=" * 80)
    print("Starting Upgraded Cadastral Boundary Correction Pipeline (Gold/Platinum Level)")
    print("=" * 80)

    # Step 1: Load village
    print("\n[1/11] Loading village data...")
    village_data = load_village(village_dir)
    print(f"   Loaded {len(village_data.plots)} plots")
    if village_data.example_truths is not None:
        print(f"   Found {len(village_data.example_truths)} example truths for calibration")

    # Step 2: Compute global alignment
    print("\n[2/11] Computing global alignment...")
    global_alignment = compute_global_alignment(village_data)
    print(f"   Global shift: dx={global_alignment.dx:.2f}m, dy={global_alignment.dy:.2f}m (method: {global_alignment.method})")

    # Initialize storage
    plot_shifts: Dict[str, PlotShift] = {}
    original_geometries: Dict[str, BaseGeometry] = {}
    corrected_geometries: Dict[str, BaseGeometry] = {}
    area_scores: Dict[str, float] = {}
    alignment_scores: Dict[str, float] = {}
    edge_scores: Dict[str, float] = {}
    confidence_inputs_dict: Dict[str, ConfidenceInputs] = {}

    # Step 3-7: Iterate plots, extract patches, detect boundaries, align, compute area/edge scores
    print("\n[3-7/11] Processing plots (extracting patches, detecting boundaries, aligning)...")
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
                buffer_meters=40.0,
            )

            # Step 4: Detect boundaries
            detection_result = detect_boundaries(patch.rgb_image, patch.boundary_mask)
            edge_scores[plot_num] = detection_result.edge_confidence

            # Step 5: Align plot with global initial guess
            alignment_result, align_score = align_plot(
                plot_geom,
                detection_result.binary_mask,
                patch.transform,
                patch.crs,
                global_shift=(global_alignment.dx, global_alignment.dy),
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

            # Store initial confidence inputs (neighbor will be updated later)
            confidence_inputs_dict[plot_num] = ConfidenceInputs(
                alignment_score=alignment_result.score,
                area_score=area_result.area_score,
                edge_score=detection_result.edge_confidence,
                neighbor_score=0.5,
            )
        except Exception as e:
            print(f"   Warning: Failed to process plot {plot_num}: {e}")
            corrected_geometries[plot_num] = plot_geom
            plot_shifts[plot_num] = PlotShift(plot_number=plot_num, dx=global_alignment.dx, dy=global_alignment.dy)
            area_scores[plot_num] = 0.0
            alignment_scores[plot_num] = 0.0
            edge_scores[plot_num] = 0.0
            confidence_inputs_dict[plot_num] = ConfidenceInputs(
                alignment_score=0.0,
                area_score=0.0,
                edge_score=0.0,
                neighbor_score=0.5,
            )

    print(f"   Processed {len(village_data.plots)}/{len(village_data.plots)} plots.")

    # Step 8: Calibrate confidence using example truths if available
    print("\n[8/11] Calibrating confidence...")
    calibrator = CalibratedConfidence()
    if village_data.example_truths is not None:
        truths = village_data.example_truths
        common_plots = [p for p in truths.index if p in confidence_inputs_dict and p in corrected_geometries]
        if len(common_plots) >= 3:
            # Compute IoU for common plots
            ious = []
            inputs_list = []
            utm_crs = _utm_crs_for(village_data.plots.loc[common_plots[0], "geometry"].centroid.x, village_data.plots.loc[common_plots[0], "geometry"].centroid.y)
            truths_utm = truths.to_crs(utm_crs)
            plots_utm = gpd.GeoDataFrame(
                {"geometry": [corrected_geometries[p] for p in common_plots]},
                index=common_plots,
                crs="EPSG:4326"
            ).to_crs(utm_crs)
            for p in common_plots:
                pred = plots_utm.loc[p, "geometry"]
                true = truths_utm.loc[p, "geometry"]
                if pred is None or true is None or pred.is_empty or true.is_empty:
                    iou = 0.0
                else:
                    union = pred.union(true).area
                    iou = pred.intersection(true).area / union if union > 0 else 0.0
                ious.append(float(iou))
                inputs_list.append(confidence_inputs_dict[p])
            calibrator.fit(inputs_list, ious)
            print(f"   Calibrated confidence using {len(common_plots)} example truths")

    # Step 9-10: Compute neighbor score, confidence, correct or flag
    print("\n[9-10/11] Computing neighbor scores, confidence, and generating predictions...")
    plot_predictions: List[PlotPrediction] = []

    for plot_num in village_data.plots.index:
        plot_geom = original_geometries[plot_num]
        neighbors = village_data.find_neighbors(plot_num, distance_m=1.0)
        neighbor_shifts = [plot_shifts[n] for n in neighbors if n in plot_shifts]

        plot_shift = plot_shifts[plot_num]
        neighbor_result = compute_neighbor_score(plot_shift, neighbor_shifts)

        inputs = confidence_inputs_dict[plot_num]
        final_inputs = ConfidenceInputs(
            alignment_score=inputs.alignment_score,
            area_score=inputs.area_score,
            edge_score=inputs.edge_score,
            neighbor_score=neighbor_result.neighbor_score,
        )

        confidence_result = compute_confidence(final_inputs, calibrator)

        prediction = create_prediction(
            plot_number=plot_num,
            confidence=confidence_result.confidence,
            corrected_geometry=corrected_geometries[plot_num],
            original_geometry=plot_geom,
            method_note=f"dx={plot_shift.dx:.2f}m, dy={plot_shift.dy:.2f}m, rot={0:.2f}deg, global={global_alignment.method}",
            corrected_threshold=0.7,
        )
        plot_predictions.append(prediction)

    # Step 11: Write predictions.geojson
    print("\n[11/11] Generating predictions GeoDataFrame and saving...")
    predictions_gdf = generate_predictions_gdf(plot_predictions)
    write_predictions(output_path, predictions_gdf)

    # Print summary
    print("\n" + "=" * 80)
    print("Upgraded Pipeline Complete! Summary:")
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
