#!/usr/bin/env python3
"""
Vision-pipeline robustness test: generates synthetic eye-to-hand scenes
(known ground-truth object position/orientation/color on a workspace-like
background), runs them through the actual ObjectDetector/HomographyMapper
pipeline in vision_pipeline.py under systematically varied illumination,
contrast, and pixel noise, and measures real detection accuracy -- this
directly exercises the OpenCV pipeline described in the paper (HSV
segmentation, morphology, contour/minAreaRect, homography), rather than
asserting robustness claims analytically.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from vision_pipeline import HomographyMapper, ObjectDetector

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

W, H = 1280, 720
BGR_COLORS = {
    "red": (40, 40, 200), "blue": (200, 60, 40), "green": (50, 160, 60), "yellow": (40, 210, 220),
}


def make_scene(color_name, cx, cy, size, angle_deg, brightness=1.0, contrast=1.0, noise_std=0.0, seed=0):
    """Draws a rotated square of the given color at (cx, cy) on a grey workspace background."""
    rng = np.random.RandomState(seed)
    img = np.full((H, W, 3), (120, 120, 118), dtype=np.uint8)  # neutral workspace background

    rect = ((cx, cy), (size, size), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], BGR_COLORS[color_name])

    img = img.astype(np.float32)
    img = img * contrast * brightness
    if noise_std > 0:
        img += rng.normal(0, noise_std, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def build_identity_mapper():
    """Homography mapper calibrated with a simple, known pixel<->robot correspondence
    (a 1:1000 px->mm affine-like mapping) so ground-truth error can be computed exactly."""
    mapper = HomographyMapper()
    image_pts = np.array([[100, 100], [1180, 100], [1180, 620], [100, 620]], dtype=np.float64)
    robot_pts = np.array([[0.0, 0.0], [0.6, 0.0], [0.6, 0.4], [0.0, 0.4]], dtype=np.float64)  # 0.6 x 0.4 m workspace
    mapper.calibrate_from_points(image_pts, robot_pts)
    return mapper


def run_condition(factor_name, level_value, n_trials, brightness=1.0, contrast=1.0, noise_std=0.0, seed0=0):
    mapper = build_identity_mapper()
    detector = ObjectDetector(mapper)
    rng = np.random.RandomState(seed0)

    detected_count = 0
    centroid_errors_px = []
    orientation_errors_deg = []

    for trial in range(n_trials):
        color = rng.choice(["red", "blue", "green", "yellow"])
        cx = rng.uniform(300, 980)
        cy = rng.uniform(200, 520)
        size = rng.uniform(60, 100)
        angle_gt = rng.uniform(-44, 44)

        img = make_scene(color, cx, cy, size, angle_gt, brightness, contrast, noise_std, seed=seed0 * 1000 + trial)
        detections = detector.process_frame(img, min_area=800.0)
        matches = [d for d in detections if d.color == color]

        if not matches:
            continue
        # pick the detection closest to ground truth (handles rare spurious contours)
        best = min(matches, key=lambda d: np.hypot(d.centroid_pixel[0] - cx, d.centroid_pixel[1] - cy))
        detected_count += 1
        centroid_errors_px.append(float(np.hypot(best.centroid_pixel[0] - cx, best.centroid_pixel[1] - cy)))
        # orientation error, accounting for the +/-45 deg wrap ambiguity of minAreaRect
        err = abs(best.orientation_deg - angle_gt)
        err = min(err, abs(err - 90))
        orientation_errors_deg.append(float(err))

    return {
        "factor": factor_name, "level": level_value, "n_trials": n_trials,
        "detection_rate": detected_count / n_trials,
        "mean_centroid_error_px": float(np.mean(centroid_errors_px)) if centroid_errors_px else float("nan"),
        "mean_orientation_error_deg": float(np.mean(orientation_errors_deg)) if orientation_errors_deg else float("nan"),
    }


def main():
    n_trials = 40
    records = []

    # Illumination sweep (brightness multiplier)
    for i, b in enumerate([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]):
        records.append(run_condition("brightness", b, n_trials, brightness=b, seed0=100 + i))

    # Contrast sweep
    for i, c in enumerate([0.4, 0.6, 0.8, 1.0, 1.2]):
        records.append(run_condition("contrast", c, n_trials, contrast=c, seed0=200 + i))

    # Pixel (camera) noise sweep -- proxy for sensor/camera-error robustness
    for i, ns in enumerate([0.0, 5.0, 10.0, 20.0, 35.0, 50.0]):
        records.append(run_condition("pixel_noise_std", ns, n_trials, noise_std=ns, seed0=300 + i))

    df = pd.DataFrame(records)
    out_path = os.path.join(RESULTS, "vision_robustness.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved {len(df)} condition summaries to {out_path}")


if __name__ == "__main__":
    main()
