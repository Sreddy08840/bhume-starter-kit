"""
Field boundary detection module using computer vision techniques.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


@dataclass
class BoundaryDetectionResult:
    """
    Result of field boundary detection.

    Attributes:
        binary_mask: Binary boundary mask (H, W) - 1 for boundaries, 0 otherwise
        edge_confidence: Confidence score [0, 1] indicating quality of detected edges
        contours: List of detected contours (each is (N, 1, 2) numpy array)
    """
    binary_mask: np.ndarray
    edge_confidence: float
    contours: list[np.ndarray]


class FieldBoundaryDetector:
    """
    Detects field boundaries in satellite image patches.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel_size: int = 5,
        morph_kernel_size: int = 3,
    ):
        """
        Initialize the field boundary detector.

        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            morph_kernel_size: Size of morphological operation kernel (must be odd)
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size

    def detect(
        self,
        rgb_image: np.ndarray,
        boundary_hint: Optional[np.ndarray] = None,
    ) -> BoundaryDetectionResult:
        """
        Detect field boundaries in an RGB image patch.

        Args:
            rgb_image: Input RGB image as (H, W, 3) uint8 numpy array
            boundary_hint: Optional boundary mask hint (e.g., from boundaries.tif)

        Returns:
            BoundaryDetectionResult containing binary mask, confidence, and contours
        """
        # Step 1: Convert RGB to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        )

        # Step 3: Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Step 4: Incorporate boundary hint if available
        if boundary_hint is not None:
            # Normalize boundary hint to [0, 255]
            hint_normalized = cv2.normalize(
                boundary_hint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            # Weighted fusion of edges and hint
            edges = cv2.addWeighted(edges, 0.7, hint_normalized, 0.3, 0)

        # Step 5: Morphological operations to clean up edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        # Close small gaps in edges
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # Thin edges to 1 pixel
        thinned = cv2.ximgproc.thinning(closed) if hasattr(cv2, 'ximgproc') else closed

        # Step 6: Extract contours
        contours, _ = cv2.findContours(
            thinned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 7: Create binary mask
        binary_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(binary_mask, contours, -1, 1, 1)

        # Step 8: Calculate edge confidence score
        edge_confidence = self._calculate_confidence(thinned, gray)

        return BoundaryDetectionResult(
            binary_mask=binary_mask,
            edge_confidence=edge_confidence,
            contours=contours,
        )

    def _calculate_confidence(
        self,
        edge_map: np.ndarray,
        gray_image: np.ndarray,
    ) -> float:
        """
        Calculate confidence score for detected edges.

        Args:
            edge_map: Edge map (H, W)
            gray_image: Grayscale input image (H, W)

        Returns:
            Confidence score in [0, 1]
        """
        # Calculate edge density
        edge_density = np.sum(edge_map > 0) / (edge_map.shape[0] * edge_map.shape[1])

        # Calculate gradient magnitude along edges
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Average gradient magnitude at edge pixels
        edge_pixels = edge_map > 0
        if np.sum(edge_pixels) == 0:
            avg_grad = 0.0
        else:
            avg_grad = np.mean(grad_mag[edge_pixels])

        # Normalize average gradient
        normalized_grad = min(avg_grad / 255.0, 1.0)

        # Combine edge density and gradient strength
        # We want reasonable edge density (not too sparse, not too dense)
        density_score = 1.0 - 2.0 * abs(edge_density - 0.05)
        density_score = max(0.0, min(1.0, density_score))

        confidence = 0.6 * normalized_grad + 0.4 * density_score

        return float(confidence)
