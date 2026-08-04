#!/usr/bin/env python3
"""
Paper_Project/python/vision_pipeline.py
========================================
Eye-to-Hand Computer Vision & Camera Geometry Calibration Pipeline
Target Journal: International Journal of Systems Science (Taylor & Francis)

Provides:
- Pinhole Camera Intrinsic/Extrinsic Matrix & Lens Distortion Modeling
- Perspective Homography Calibration (Pixel to Robot 2D Ground Frame)
- Adaptive HSV Color Segmentation (Red, Blue, Green, Yellow Cubes)
- Morphological Filtering, Contour Analysis & Polygon Approximation
- Minimum Area Bounding Box, Centroid, & Orientation Angle Detection
- Automatic Gripper Alignment Calculation
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional

@dataclass
class DetectedObject:
    color: str
    centroid_pixel: Tuple[float, float]
    centroid_robot: Tuple[float, float]  # (X_r, Y_r) in meters
    orientation_deg: float
    gripper_angle_deg: float
    bbox_corners_pixel: np.ndarray
    area_pixels: float
    confidence: float

class CameraCalibrator:
    def __init__(self, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        """
        Pinhole Camera Intrinsic Matrix K and Lens Distortion Coefficients.
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        dist = [k1, k2, p1, p2, k3]
        """
        if camera_matrix is not None:
            self.K = camera_matrix
        else:
            # Default calibrated USB camera intrinsics (1280x720)
            self.K = np.array([
                [920.0,   0.0, 640.0],
                [  0.0, 920.0, 360.0],
                [  0.0,   0.0,   1.0]
            ], dtype=np.float64)
            
        if dist_coeffs is not None:
            self.dist = dist_coeffs
        else:
            self.dist = np.array([-0.05, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies radial and tangential lens distortion correction."""
        return cv2.undistort(frame, self.K, self.dist)


class HomographyMapper:
    def __init__(self):
        """
        Planar Homography Matrix H (3x3) mapping image pixels (u, v, 1)^T to robot plane (X_r, Y_r, 1)^T.
        s * [X_r, Y_r, 1]^T = H * [u, v, 1]^T
        """
        self.H = np.eye(3, dtype=np.float64)
        self.H_inv = np.eye(3, dtype=np.float64)
        self.is_calibrated = False

    def calibrate_from_points(self, image_pts: np.ndarray, robot_pts: np.ndarray) -> float:
        """
        Computes Homography matrix using Direct Linear Transform (DLT) with RANSAC.
        :param image_pts: (N, 2) pixel coordinates [[u1, v1], [u2, v2], ...]
        :param robot_pts: (N, 2) robot world coordinates in meters [[X1, Y1], [X2, Y2], ...]
        :return: Reprojection RMSE error in meters.
        """
        assert len(image_pts) >= 4 and len(robot_pts) >= 4, "Minimum 4 point correspondences required."
        self.H, mask = cv2.findHomography(image_pts, robot_pts, cv2.RANSAC, 5.0)
        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True

        # Calculate Reprojection Error
        reprojected = self.pixel_to_robot(image_pts)
        rmse = float(np.sqrt(np.mean((reprojected - robot_pts)**2)))
        return rmse

    def pixel_to_robot(self, pixels: np.ndarray) -> np.ndarray:
        """Maps array of pixel coordinates (N, 2) to robot world coordinates (N, 2) in meters."""
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
        
        pts_homo = np.hstack([pixels, np.ones((len(pixels), 1))])  # (N, 3)
        robot_homo = (self.H @ pts_homo.T).T                        # (N, 3)
        robot_pts = robot_homo[:, :2] / robot_homo[:, 2:]           # normalize
        return robot_pts

    def robot_to_pixel(self, robot_pts: np.ndarray) -> np.ndarray:
        """Maps array of robot world coordinates (N, 2) to pixel coordinates (N, 2)."""
        if robot_pts.ndim == 1:
            robot_pts = robot_pts.reshape(1, 2)
            
        pts_homo = np.hstack([robot_pts, np.ones((len(robot_pts), 1))])
        pixel_homo = (self.H_inv @ pts_homo.T).T
        pixels = pixel_homo[:, :2] / pixel_homo[:, 2:]
        return pixels


class ObjectDetector:
    def __init__(self, homography_mapper: HomographyMapper):
        self.mapper = homography_mapper

        # HSV Color Range Definitions
        self.hsv_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 150, 70]), np.array([130, 255, 255]))
            ],
            'green': [
                (np.array([35, 100, 70]), np.array([85, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ]
        }

    def process_frame(self, frame: np.ndarray, min_area: float = 800.0) -> List[DetectedObject]:
        """
        Full computer vision pipeline:
        1. Convert BGR to HSV color space.
        2. Create color threshold masks and apply Morphological Opening/Closing.
        3. Contour analysis, area filtering, and minimum bounding rectangle fit.
        4. Orientation calculation & pixel-to-robot homography projection.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        detected_objects = []

        for color_name, ranges in self.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            # Morphological filtering for noise removal
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Contour extraction
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                # Moments for centroid
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                u_c = M["m10"] / M["m00"]
                v_c = M["m01"] / M["m00"]

                # Minimum Area Rectangle fit for orientation
                rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
                (cx, cy), (width, height), angle = rect

                # Normalize orientation angle to [-45, 45] degrees
                if width < height:
                    angle_norm = angle - 90.0
                else:
                    angle_norm = angle
                
                # Wrap to [-45, 45]
                while angle_norm > 45.0:
                    angle_norm -= 90.0
                while angle_norm < -45.0:
                    angle_norm += 90.0

                # Calculate Gripper Rotation Angle (opposite correction)
                gripper_angle = -angle_norm

                # Project centroid to robot workspace coordinates (meters)
                robot_pos = self.mapper.pixel_to_robot(np.array([[u_c, v_c]]))[0]

                # Bounding box corners
                box_corners = cv2.boxPoints(rect)

                obj = DetectedObject(
                    color=color_name,
                    centroid_pixel=(u_c, v_c),
                    centroid_robot=(robot_pos[0], robot_pos[1]),
                    orientation_deg=angle_norm,
                    gripper_angle_deg=gripper_angle,
                    bbox_corners_pixel=box_corners,
                    area_pixels=area,
                    confidence=min(1.0, area / 3500.0)
                )
                detected_objects.append(obj)

        return detected_objects
