from __future__ import annotations

import math

FAMILIES = ["classic", "v_shaped", "split", "y_shaped", "mesh"]
SPOKE_COUNTS = list(range(5, 25))
CENTER_TYPES = ["open", "cap"]
FINISHES = ["solid_silver", "solid_black", "diamond_cut"]
DESIGN_VARIANTS = ["straight", "swept", "wide", "slim", "angular", "turbine", "petal", "concave"]
REGRESSION_FIELDS = ["center_hole_ratio", "bulge_depth_ratio", "face_thickness_ratio", "rim_depth_ratio"]
REGRESSION_SOURCE_FIELDS = ["center_hole_diameter_mm", "bulge_depth_mm", "face_thickness_mm", "rim_depth_mm"]
RAW_FIELD_RANGES = {"outer_diameter_mm": (330.0, 460.0), "center_hole_diameter_mm": (50.0, 74.0), "bulge_depth_mm": (8.0, 28.0), "face_thickness_mm": (5.0, 11.0), "rim_depth_mm": (90.0, 170.0)}
REGRESSION_RANGES = {"center_hole_ratio": (50.0 / 460.0, 0.19), "bulge_depth_ratio": (8.0 / 460.0, 28.0 / 330.0), "face_thickness_ratio": (5.0 / 460.0, 11.0 / 330.0), "rim_depth_ratio": (90.0 / 460.0, 170.0 / 330.0)}
SCHEMA_VERSION = 2
GENERATOR_VERSION = "4.0.0"
IMAGE_SIZE = 320
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def ratios_from_row(row: dict) -> dict[str, float]:
    diameter = float(row["outer_diameter_mm"])
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("outer_diameter_mm должен быть положительным числом")
    return {name: float(row[source]) / diameter for name, source in zip(REGRESSION_FIELDS, REGRESSION_SOURCE_FIELDS)}

def normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    result = {}
    for name, (lo, hi) in REGRESSION_RANGES.items():
        value = float(ratios[name])
        result[name] = max(-1.0, min(1.0, 2 * (value - lo) / (hi - lo) - 1))
    return result

def denormalize_ratios(values: dict[str, float]) -> dict[str, float]:
    return {name: lo + (max(-1.0, min(1.0, float(values[name]))) + 1) * (hi - lo) / 2 for name, (lo, hi) in REGRESSION_RANGES.items()}