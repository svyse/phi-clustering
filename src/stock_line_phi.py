"""
Stock line pixel-grid clustering with J(x, y) and higher-dimensional lifting.

This module implements a complete project pipeline:

1. Generate a complex two-line stock-like chart.
2. Treat the chart as a pixel grid with origin at the axes intersection.
3. Build x-axis and y-axis point sets: J(x, 0) and J(0, y).
4. Build line point sets and compute:
       Phi(x, y) = [x, y, slope, curvature, J, F, H]
5. Cluster line points in Phi-space, then project the cluster labels back to 2D.
6. Detect intersections where both lines have the same J value at the same x.
7. Represent whole lines and axes by J-value sequences and cluster those vectors.

The default J is the chart-height value:

       J(x, y) = y

For a line chart this is practical because two lines intersect at a shared x
when their J values are equal.
"""

from __future__ import annotations

import os

# Limit BLAS/OpenMP threads so KMeans and MDS run reliably in notebooks and
# hosted environments with small CPU allocations.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PIL import Image, ImageDraw

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - project still works without plotly
    go = None


@dataclass
class ChartConfig:
    """Configuration for the generated chart and feature pipeline."""

    seed: int = 42
    canvas_width: int = 1200
    canvas_height: int = 800
    plot_left: int = 90
    plot_bottom: int = 700
    plot_width: int = 1040
    plot_height: int = 620
    line_width: int = 4
    axis_width: int = 4
    grid_step: int = 100
    vector_length: int = 240
    intersection_tolerance_px: float = 2.0
    slope_lift_scale: float = 42.0
    curvature_lift_scale: float = 850.0
    vertical_axis_slope_value: float = 10.0
    object_lift_scale: float = 10000.0
    k_known_lines: int = 2
    k_point_objects: int = 4
    lifting_distinctness_threshold: float = 2.0
    lifting_silhouette_threshold: float = 0.20

    color_background: Tuple[int, int, int] = (255, 255, 255)
    color_plot_background: Tuple[int, int, int] = (250, 250, 250)
    color_grid: Tuple[int, int, int] = (224, 224, 224)
    color_axis: Tuple[int, int, int] = (0, 0, 0)
    color_line_1: Tuple[int, int, int] = (220, 20, 60)
    color_line_2: Tuple[int, int, int] = (30, 144, 255)
    color_intersection: Tuple[int, int, int] = (255, 165, 0)


# -----------------------------------------------------------------------------
# Coordinate and feature functions
# -----------------------------------------------------------------------------


def grid_to_image_xy(x_grid: np.ndarray, y_grid: np.ndarray, cfg: ChartConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert chart-grid coordinates to image pixel coordinates.

    The chart grid uses a mathematical orientation where y increases upward.
    The image uses a raster orientation where pixel rows increase downward.

        pixel_x = plot_left + x_grid
        pixel_y = plot_bottom - y_grid
    """

    pixel_x = cfg.plot_left + np.asarray(x_grid)
    pixel_y = cfg.plot_bottom - np.asarray(y_grid)
    return pixel_x, pixel_y


def image_to_grid_xy(pixel_x: np.ndarray, pixel_y: np.ndarray, cfg: ChartConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image pixel coordinates back to chart-grid coordinates."""

    x_grid = np.asarray(pixel_x) - cfg.plot_left
    y_grid = cfg.plot_bottom - np.asarray(pixel_y)
    return x_grid, y_grid


def compute_J(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Compute J(x, y).

    Default line-chart interpretation:

        J(x, y) = y

    This makes J the plotted chart height or value at that pixel coordinate.
    It also gives a simple intersection rule for two lines at the same x:

        if J_1(x) == J_2(x), then the two lines have the same y value.
    """

    _ = x_grid
    return np.asarray(y_grid, dtype=float)


def compute_slope_and_curvature(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local slope and curvature for a line chart sequence.

    slope is dy/dx in pixel units.
    curvature is the standard planar-curve curvature approximation:

        curvature = y_second / (1 + slope^2)^(3/2)
    """

    y_float = np.asarray(y, dtype=float)
    slope = np.gradient(y_float)
    second = np.gradient(slope)
    curvature = second / np.power(1.0 + slope ** 2, 1.5)
    return slope, curvature


def compute_lifted_features(df: pd.DataFrame, cfg: ChartConfig) -> pd.DataFrame:
    """Add J, F, H, and Phi-ready columns to a point table.

    Practical feature map for line charts:

        Phi(x, y) = [x, y, slope, curvature, J, F, H]

    with:

        J = y
        F = J + slope_lift_scale * slope
        H = F + curvature_lift_scale * curvature

    F and H are simple higher-dimensional lifts. They are intentionally easy to
    replace with a different feature map later.
    """

    out = df.copy()
    out["J"] = compute_J(out["x_grid"].to_numpy(), out["y_grid"].to_numpy())
    out["F"] = out["J"] + cfg.slope_lift_scale * out["slope"]
    out["H"] = out["F"] + cfg.curvature_lift_scale * out["curvature"]
    return out


# -----------------------------------------------------------------------------
# Synthetic chart generation
# -----------------------------------------------------------------------------


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _normalize_jointly(a: np.ndarray, b: np.ndarray, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([a, b])
    mn = float(combined.min())
    mx = float(combined.max())
    if np.isclose(mx, mn):
        midpoint = (low + high) / 2.0
        return np.full_like(a, midpoint), np.full_like(b, midpoint)
    a_scaled = low + (a - mn) * (high - low) / (mx - mn)
    b_scaled = low + (b - mn) * (high - low) / (mx - mn)
    return a_scaled, b_scaled


def generate_stock_like_lines(cfg: ChartConfig) -> pd.DataFrame:
    """Generate two complex stock-like lines directly on the pixel grid."""

    rng = np.random.default_rng(cfg.seed)
    x = np.arange(cfg.plot_width + 1, dtype=float)

    walk = np.cumsum(rng.normal(0.0, 1.0, size=len(x)))
    walk = _moving_average(walk, window=23)

    base = (
        300.0
        + 0.035 * x
        + 55.0 * np.sin(x / 45.0)
        + 36.0 * np.sin(x / 19.0 + 0.7)
        + 45.0 * np.cos(x / 130.0)
        + 1.9 * walk
    )

    spread = (
        95.0 * np.sin(x / 82.0)
        + 45.0 * np.sin(x / 29.0 + 1.1)
        + 25.0 * np.cos(x / 141.0)
    )

    noise_1 = _moving_average(rng.normal(0.0, 9.0, size=len(x)), window=7)
    noise_2 = _moving_average(rng.normal(0.0, 9.0, size=len(x)), window=7)

    y1_raw = base + 0.50 * spread + noise_1
    y2_raw = base - 0.58 * spread + noise_2

    margin = 55.0
    y1, y2 = _normalize_jointly(y1_raw, y2_raw, low=margin, high=cfg.plot_height - margin)
    y1 = np.rint(y1).astype(int)
    y2 = np.rint(y2).astype(int)

    slope_1, curvature_1 = compute_slope_and_curvature(y1)
    slope_2, curvature_2 = compute_slope_and_curvature(y2)

    return pd.DataFrame(
        {
            "x_grid": x.astype(int),
            "line_1_y": y1,
            "line_2_y": y2,
            "line_1_slope": slope_1,
            "line_2_slope": slope_2,
            "line_1_curvature": curvature_1,
            "line_2_curvature": curvature_2,
        }
    )


# -----------------------------------------------------------------------------
# Raster rendering and mask extraction
# -----------------------------------------------------------------------------


def _draw_polyline_mask(points_image: Iterable[Tuple[int, int]], size: Tuple[int, int], width: int) -> np.ndarray:
    """Rasterize a polyline into a boolean mask."""

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    pts = [(int(x), int(y)) for x, y in points_image]
    if len(pts) >= 2:
        draw.line(pts, fill=255, width=width, joint="curve")
    return np.asarray(mask) > 0


def render_chart(lines: pd.DataFrame, intersections: pd.DataFrame, cfg: ChartConfig, output_path: Path) -> Dict[str, np.ndarray]:
    """Render a chart image and return boolean masks for axes and lines."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = (cfg.canvas_width, cfg.canvas_height)

    img = Image.new("RGB", size, cfg.color_background)
    draw = ImageDraw.Draw(img)

    plot_right = cfg.plot_left + cfg.plot_width
    plot_top = cfg.plot_bottom - cfg.plot_height

    # Plot area and grid.
    draw.rectangle([cfg.plot_left, plot_top, plot_right, cfg.plot_bottom], fill=cfg.color_plot_background)
    for x in range(0, cfg.plot_width + 1, cfg.grid_step):
        px = cfg.plot_left + x
        draw.line([(px, plot_top), (px, cfg.plot_bottom)], fill=cfg.color_grid, width=1)
    for y in range(0, cfg.plot_height + 1, cfg.grid_step):
        py = cfg.plot_bottom - y
        draw.line([(cfg.plot_left, py), (plot_right, py)], fill=cfg.color_grid, width=1)

    # Axis masks.
    x_axis_points = [(cfg.plot_left, cfg.plot_bottom), (plot_right, cfg.plot_bottom)]
    y_axis_points = [(cfg.plot_left, cfg.plot_bottom), (cfg.plot_left, plot_top)]
    x_axis_mask = _draw_polyline_mask(x_axis_points, size, cfg.axis_width)
    y_axis_mask = _draw_polyline_mask(y_axis_points, size, cfg.axis_width)

    # Line masks. These are kept separate so intersections are not lost if one
    # rendered line would visually overwrite the other.
    x = lines["x_grid"].to_numpy()
    line1_x, line1_y_img = grid_to_image_xy(x, lines["line_1_y"].to_numpy(), cfg)
    line2_x, line2_y_img = grid_to_image_xy(x, lines["line_2_y"].to_numpy(), cfg)
    line1_pts = list(zip(line1_x.astype(int), line1_y_img.astype(int)))
    line2_pts = list(zip(line2_x.astype(int), line2_y_img.astype(int)))
    line1_mask = _draw_polyline_mask(line1_pts, size, cfg.line_width)
    line2_mask = _draw_polyline_mask(line2_pts, size, cfg.line_width)
    intersection_mask = line1_mask & line2_mask

    # Compose the visible image from masks.
    arr = np.asarray(img).copy()
    arr[line1_mask] = np.array(cfg.color_line_1, dtype=np.uint8)
    arr[line2_mask] = np.array(cfg.color_line_2, dtype=np.uint8)
    arr[intersection_mask] = np.array(cfg.color_intersection, dtype=np.uint8)
    arr[x_axis_mask | y_axis_mask] = np.array(cfg.color_axis, dtype=np.uint8)

    # Add a simple title and labels outside the plot area.
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.text((cfg.plot_left, 22), "Synthetic stock-like chart: two complex lines", fill=(30, 30, 30))
    draw.text((cfg.plot_left + cfg.plot_width // 2 - 70, cfg.plot_bottom + 32), "x pixel grid", fill=(30, 30, 30))
    draw.text((8, plot_top + cfg.plot_height // 2), "y pixel grid", fill=(30, 30, 30))

    # Draw intersection markers calculated from equal J values.
    if not intersections.empty:
        for _, row in intersections.iterrows():
            px, py = grid_to_image_xy(np.array([row["x_grid"]]), np.array([row["J"]]), cfg)
            cx = int(round(float(px[0])))
            cy = int(round(float(py[0])))
            r = 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=cfg.color_intersection, width=3)

    img.save(output_path)

    return {
        "line_1_mask": line1_mask,
        "line_2_mask": line2_mask,
        "x_axis_mask": x_axis_mask,
        "y_axis_mask": y_axis_mask,
        "intersection_mask": intersection_mask,
    }


def mask_to_grid_points(mask: np.ndarray, cfg: ChartConfig) -> pd.DataFrame:
    """Convert a boolean image mask into pixel and chart-grid coordinates."""

    rows, cols = np.where(mask)
    x_grid, y_grid = image_to_grid_xy(cols, rows, cfg)
    in_plot = (x_grid >= 0) & (x_grid <= cfg.plot_width) & (y_grid >= 0) & (y_grid <= cfg.plot_height)
    return pd.DataFrame(
        {
            "pixel_x": cols[in_plot].astype(int),
            "pixel_y": rows[in_plot].astype(int),
            "x_grid": x_grid[in_plot].astype(int),
            "y_grid": y_grid[in_plot].astype(int),
        }
    )


def extract_raster_pixels_from_masks(masks: Dict[str, np.ndarray], cfg: ChartConfig) -> pd.DataFrame:
    """Extract all raster pixels from the line and axis masks."""

    frames = []
    for key, object_id, object_kind in [
        ("line_1_mask", "line_1", "chart_line"),
        ("line_2_mask", "line_2", "chart_line"),
        ("x_axis_mask", "x_axis_J_x_0", "axis"),
        ("y_axis_mask", "y_axis_J_0_y", "axis"),
    ]:
        df = mask_to_grid_points(masks[key], cfg)
        df["object_id"] = object_id
        df["object_kind"] = object_kind
        frames.append(df)
    pixels = pd.concat(frames, ignore_index=True)
    pixels["J"] = compute_J(pixels["x_grid"].to_numpy(), pixels["y_grid"].to_numpy())
    pixels["is_x_axis_formula"] = pixels["y_grid"].eq(0)
    pixels["is_y_axis_formula"] = pixels["x_grid"].eq(0)
    return pixels


# -----------------------------------------------------------------------------
# Centerline feature table
# -----------------------------------------------------------------------------


def build_centerline_feature_table(lines: pd.DataFrame, cfg: ChartConfig) -> pd.DataFrame:
    """Create centerline features for the two chart lines plus axes."""

    frames: List[pd.DataFrame] = []

    for object_id, y_col, slope_col, curv_col in [
        ("line_1", "line_1_y", "line_1_slope", "line_1_curvature"),
        ("line_2", "line_2_y", "line_2_slope", "line_2_curvature"),
    ]:
        x_grid = lines["x_grid"].to_numpy()
        y_grid = lines[y_col].to_numpy()
        pixel_x, pixel_y = grid_to_image_xy(x_grid, y_grid, cfg)
        df = pd.DataFrame(
            {
                "object_id": object_id,
                "object_kind": "chart_line",
                "x_grid": x_grid,
                "y_grid": y_grid,
                "pixel_x": pixel_x.astype(int),
                "pixel_y": pixel_y.astype(int),
                "slope": lines[slope_col].to_numpy(dtype=float),
                "curvature": lines[curv_col].to_numpy(dtype=float),
            }
        )
        frames.append(df)

    # x-axis as a centerline object: y = 0, slope = 0, curvature = 0.
    x_axis_x = np.arange(cfg.plot_width + 1)
    x_axis_y = np.zeros_like(x_axis_x)
    px, py = grid_to_image_xy(x_axis_x, x_axis_y, cfg)
    frames.append(
        pd.DataFrame(
            {
                "object_id": "x_axis_J_x_0",
                "object_kind": "axis",
                "x_grid": x_axis_x,
                "y_grid": x_axis_y,
                "pixel_x": px.astype(int),
                "pixel_y": py.astype(int),
                "slope": np.zeros_like(x_axis_x, dtype=float),
                "curvature": np.zeros_like(x_axis_x, dtype=float),
            }
        )
    )

    # y-axis as a centerline object: x = 0. A vertical line has infinite slope,
    # so we use a finite clipped slope value for clustering and visualization.
    y_axis_y = np.arange(cfg.plot_height + 1)
    y_axis_x = np.zeros_like(y_axis_y)
    px, py = grid_to_image_xy(y_axis_x, y_axis_y, cfg)
    frames.append(
        pd.DataFrame(
            {
                "object_id": "y_axis_J_0_y",
                "object_kind": "axis",
                "x_grid": y_axis_x,
                "y_grid": y_axis_y,
                "pixel_x": px.astype(int),
                "pixel_y": py.astype(int),
                "slope": np.full_like(y_axis_y, cfg.vertical_axis_slope_value, dtype=float),
                "curvature": np.zeros_like(y_axis_y, dtype=float),
            }
        )
    )

    center = pd.concat(frames, ignore_index=True)
    center = compute_lifted_features(center, cfg)
    center["is_x_axis_formula"] = center["y_grid"].eq(0)
    center["is_y_axis_formula"] = center["x_grid"].eq(0)
    return center


# -----------------------------------------------------------------------------
# Intersection detection
# -----------------------------------------------------------------------------


def detect_equal_j_intersections(lines: pd.DataFrame, cfg: ChartConfig) -> pd.DataFrame:
    """Detect intersections where both lines have the same J at the same x.

    Because J(x, y) = y, equal J means equal chart height.
    The code also catches sign changes between adjacent pixels and interpolates
    the crossing location.
    """

    x = lines["x_grid"].to_numpy(dtype=float)
    j1 = lines["line_1_y"].to_numpy(dtype=float)
    j2 = lines["line_2_y"].to_numpy(dtype=float)
    diff = j1 - j2

    records: List[Dict[str, float | str]] = []

    # Near-equality at an integer x pixel.
    near = np.where(np.abs(diff) <= cfg.intersection_tolerance_px)[0]
    for i in near:
        records.append(
            {
                "x_grid": float(x[i]),
                "J": float((j1[i] + j2[i]) / 2.0),
                "line_1_J": float(j1[i]),
                "line_2_J": float(j2[i]),
                "abs_J_difference": float(abs(diff[i])),
                "method": "equal_J_tolerance",
            }
        )

    # Sign changes between adjacent x pixels.
    signs = diff[:-1] * diff[1:]
    crossing_indices = np.where(signs < 0)[0]
    for i in crossing_indices:
        d0 = diff[i]
        d1 = diff[i + 1]
        frac = abs(d0) / (abs(d0) + abs(d1)) if not np.isclose(d0, d1) else 0.5
        x_cross = x[i] + frac * (x[i + 1] - x[i])
        j1_cross = j1[i] + frac * (j1[i + 1] - j1[i])
        j2_cross = j2[i] + frac * (j2[i + 1] - j2[i])
        records.append(
            {
                "x_grid": float(x_cross),
                "J": float((j1_cross + j2_cross) / 2.0),
                "line_1_J": float(j1_cross),
                "line_2_J": float(j2_cross),
                "abs_J_difference": float(abs(j1_cross - j2_cross)),
                "method": "sign_change_interpolation",
            }
        )

    if not records:
        return pd.DataFrame(columns=["x_grid", "J", "line_1_J", "line_2_J", "abs_J_difference", "method"])

    intersections = pd.DataFrame(records).sort_values("x_grid").reset_index(drop=True)

    # Merge duplicate detections that are only a few pixels apart.
    merged: List[pd.Series] = []
    current_group: List[pd.Series] = []
    last_x = None
    for _, row in intersections.iterrows():
        if last_x is None or abs(float(row["x_grid"]) - last_x) <= 4.0:
            current_group.append(row)
        else:
            group_df = pd.DataFrame(current_group)
            best = group_df.loc[group_df["abs_J_difference"].idxmin()]
            merged.append(best)
            current_group = [row]
        last_x = float(row["x_grid"])
    if current_group:
        group_df = pd.DataFrame(current_group)
        best = group_df.loc[group_df["abs_J_difference"].idxmin()]
        merged.append(best)

    return pd.DataFrame(merged).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Clustering in Phi-space and dimensional lifting
# -----------------------------------------------------------------------------


PHI_FEATURES = ["x_grid", "y_grid", "slope", "curvature", "J", "F", "H"]

# Feature weights are applied after standardization. H receives a larger weight
# because it contains the final object-level lift from whole-line J-vector
# clustering. That is the step that makes the known line groups distinct.
FEATURE_WEIGHTS = {
    "x_grid": 0.45,
    "y_grid": 0.60,
    "slope": 0.90,
    "curvature": 0.70,
    "J": 0.60,
    "F": 0.90,
    "H": 6.00,
}

LIFTING_STAGES = [
    ("stage_01_xy", ["x_grid", "y_grid"]),
    ("stage_02_xy_J", ["x_grid", "y_grid", "J"]),
    ("stage_03_xy_J_slope", ["x_grid", "y_grid", "J", "slope"]),
    ("stage_04_xy_J_slope_curvature", ["x_grid", "y_grid", "J", "slope", "curvature"]),
    ("stage_05_xy_slope_curvature_J_F", ["x_grid", "y_grid", "slope", "curvature", "J", "F"]),
    ("stage_06_full_Phi", PHI_FEATURES),
]


def _cluster_kmeans(df: pd.DataFrame, feature_cols: List[str], k: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster rows with KMeans after feature standardization."""

    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    weights = np.array([FEATURE_WEIGHTS.get(col, 1.0) for col in feature_cols], dtype=float)
    X_scaled = X_scaled * weights
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = kmeans.fit_predict(X_scaled)
    return labels, X_scaled, kmeans.cluster_centers_


def _distinctness_ratio(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    """Compute a simple center-distance / within-spread separability ratio."""

    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    centers = np.array([X_scaled[labels == label].mean(axis=0) for label in unique])
    center_distances = pairwise_distances(centers)
    center_distance = float(center_distances[np.triu_indices_from(center_distances, k=1)].mean())
    within = []
    for label, center in zip(unique, centers):
        pts = X_scaled[labels == label]
        if len(pts):
            within.append(np.linalg.norm(pts - center, axis=1).mean())
    within_mean = float(np.mean(within)) if within else 0.0
    return center_distance / (within_mean + 1e-9)


def iterative_dimensional_lifting(line_points: pd.DataFrame, cfg: ChartConfig, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add dimensions one by one and cluster the known number of lines.

    At each stage the algorithm clusters in the stage feature space and then
    plots the labels back in the original x-y grid. This implements the idea:

        keep adding J, F, H, ... until known line groups become distinct.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    result = line_points.copy()
    summaries = []
    chosen_stage = None

    for stage_name, features in LIFTING_STAGES:
        labels, X_scaled, _ = _cluster_kmeans(result, features, cfg.k_known_lines, cfg.seed)
        result[f"{stage_name}_cluster"] = labels
        ratio = _distinctness_ratio(X_scaled, labels)
        if len(np.unique(labels)) == cfg.k_known_lines and len(result) > cfg.k_known_lines:
            try:
                sil = float(silhouette_score(X_scaled, labels))
            except Exception:
                sil = float("nan")
        else:
            sil = float("nan")

        is_distinct = bool(
            len(np.unique(labels)) == cfg.k_known_lines
            and ratio >= cfg.lifting_distinctness_threshold
            and (np.isnan(sil) or sil >= cfg.lifting_silhouette_threshold)
        )
        if chosen_stage is None and is_distinct:
            chosen_stage = stage_name

        summaries.append(
            {
                "stage": stage_name,
                "features": ",".join(features),
                "n_features": len(features),
                "n_clusters_found": int(len(np.unique(labels))),
                "distinctness_ratio": ratio,
                "silhouette_score": sil,
                "is_distinct_enough": is_distinct,
            }
        )

        save_stage_projection_plot(
            result,
            cluster_col=f"{stage_name}_cluster",
            title=f"{stage_name}: clustered in {len(features)}D, projected back to x-y",
            output_path=output_dir / f"{stage_name}_projected_to_xy.png",
        )

    result["chosen_lifting_stage"] = chosen_stage or LIFTING_STAGES[-1][0]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "dimensional_lifting_summary.csv", index=False)
    return result, summary_df


def cluster_full_phi(line_points: pd.DataFrame, cfg: ChartConfig) -> pd.DataFrame:
    """Cluster chart-line points using the full Phi feature map."""

    result = line_points.copy()
    labels, _, _ = _cluster_kmeans(result, PHI_FEATURES, cfg.k_known_lines, cfg.seed)
    result["phi_cluster"] = labels
    result["phi_cluster_name"] = result["phi_cluster"].map(lambda v: f"phi_line_cluster_{int(v)}")
    return result


# -----------------------------------------------------------------------------
# Whole-object J-vector clustering
# -----------------------------------------------------------------------------


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, values.astype(float))


def build_object_j_vectors(lines: pd.DataFrame, cfg: ChartConfig) -> pd.DataFrame:
    """Represent each whole line/axis as a fixed-length sequence of J values."""

    vectors = {
        "line_1": _resample(lines["line_1_y"].to_numpy(dtype=float), cfg.vector_length),
        "line_2": _resample(lines["line_2_y"].to_numpy(dtype=float), cfg.vector_length),
        "x_axis_J_x_0": np.zeros(cfg.vector_length, dtype=float),
        "y_axis_J_0_y": np.linspace(0.0, cfg.plot_height, cfg.vector_length),
    }
    rows = []
    for object_id, vector in vectors.items():
        row = {"object_id": object_id}
        for i, value in enumerate(vector):
            row[f"J_{i:03d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_object_j_vectors(vectors: pd.DataFrame, cfg: ChartConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster whole objects by their J-value vectors.

    The feature vector combines J, first differences, and second differences so
    the clustering can consider level, slope, and curvature of the whole object.
    """

    object_ids = vectors["object_id"].to_list()
    j_cols = [c for c in vectors.columns if c.startswith("J_")]
    raw = vectors[j_cols].to_numpy(dtype=float)
    first = np.gradient(raw, axis=1)
    second = np.gradient(first, axis=1)
    features = np.concatenate([raw, first * cfg.slope_lift_scale, second * cfg.curvature_lift_scale], axis=1)
    X = StandardScaler().fit_transform(features)

    k = min(cfg.k_point_objects, len(object_ids))
    model = KMeans(n_clusters=k, n_init=20, random_state=cfg.seed)
    labels = model.fit_predict(X)

    summary = pd.DataFrame({"object_id": object_ids, "object_vector_cluster": labels})
    distances = pairwise_distances(X)
    dist_df = pd.DataFrame(distances, index=object_ids, columns=object_ids)
    return summary, dist_df


def apply_object_vector_lift(center: pd.DataFrame, object_clusters: pd.DataFrame, cfg: ChartConfig) -> pd.DataFrame:
    """Add the whole-object vector-cluster lift into H.

    The local lifts are:

        J = y
        F = J + slope_lift_scale * slope
        H_local = F + curvature_lift_scale * curvature

    The whole-line/axis J-vector clustering supplies an object-level discrete
    lift. This is the practical "keep adding dimensions until the known number
    of groups is distinct" step. H becomes:

        H = H_local + object_lift_scale * object_vector_cluster

    This makes whole objects separable in the final Phi-space while still
    preserving the original x, y, slope, curvature, J, and F values.
    """

    out = center.copy()
    cluster_map = object_clusters.set_index("object_id")["object_vector_cluster"].to_dict()
    out["object_vector_cluster"] = out["object_id"].map(cluster_map).astype(int)
    out["H_local"] = out["H"]
    out["sequence_lift"] = cfg.object_lift_scale * out["object_vector_cluster"]
    out["H"] = out["H_local"] + out["sequence_lift"]
    return out


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def _object_color_map() -> Dict[str, str]:
    return {
        "line_1": "crimson",
        "line_2": "dodgerblue",
        "x_axis_J_x_0": "black",
        "y_axis_J_0_y": "dimgray",
        "intersection": "orange",
    }


def save_stage_projection_plot(df: pd.DataFrame, cluster_col: str, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6.5))
    plt.scatter(df["x_grid"], df["y_grid"], c=df[cluster_col], cmap="tab10", s=8, alpha=0.80)
    plt.title(title)
    plt.xlabel("x_grid pixel coordinate")
    plt.ylabel("y_grid pixel coordinate")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_original_chart_preview(chart_path: Path, output_path: Path) -> None:
    img = Image.open(chart_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Rendered stock-like chart")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_centerline_xy_plot(center: pd.DataFrame, intersections: pd.DataFrame, output_path: Path) -> None:
    colors = _object_color_map()
    plt.figure(figsize=(12, 7))
    for object_id, group in center.groupby("object_id"):
        plt.plot(group["x_grid"], group["y_grid"], label=object_id, linewidth=2.0, color=colors.get(object_id))
    if not intersections.empty:
        plt.scatter(intersections["x_grid"], intersections["J"], s=70, marker="X", color=colors["intersection"], label="equal-J intersections")
    plt.title("Centerlines and axes in the pixel grid")
    plt.xlabel("x_grid")
    plt.ylabel("y_grid")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_phi_cluster_projection(line_phi: pd.DataFrame, intersections: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    plt.scatter(line_phi["x_grid"], line_phi["y_grid"], c=line_phi["phi_cluster"], cmap="tab10", s=10, alpha=0.85)
    if not intersections.empty:
        plt.scatter(intersections["x_grid"], intersections["J"], s=80, marker="X", color="orange", label="equal-J intersections")
    plt.title("Full Phi-space clusters projected back to the original 2D chart")
    plt.xlabel("x_grid")
    plt.ylabel("y_grid")
    plt.grid(alpha=0.25)
    if not intersections.empty:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_axis_and_line_clusters_2d(center: pd.DataFrame, line_phi: pd.DataFrame, intersections: pd.DataFrame, output_path: Path) -> None:
    axis = center[center["object_kind"] == "axis"]
    plt.figure(figsize=(12, 7))
    for object_id, group in axis.groupby("object_id"):
        plt.plot(group["x_grid"], group["y_grid"], linewidth=2.5, label=object_id)
    plt.scatter(line_phi["x_grid"], line_phi["y_grid"], c=line_phi["phi_cluster"], cmap="tab10", s=10, alpha=0.85, label="line Phi clusters")
    if not intersections.empty:
        plt.scatter(intersections["x_grid"], intersections["J"], s=90, marker="X", color="orange", label="equal-J intersections")
    plt.title("Axes plus line clusters in the original x-y pixel grid")
    plt.xlabel("x_grid")
    plt.ylabel("y_grid")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_3d_phi_plot(center: pd.DataFrame, line_phi: pd.DataFrame, intersections: pd.DataFrame, output_path: Path) -> None:
    colors = _object_color_map()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    axis = center[center["object_kind"] == "axis"]
    for object_id, group in axis.groupby("object_id"):
        ax.plot(group["x_grid"], group["y_grid"], group["J"], label=object_id, color=colors.get(object_id), linewidth=2.2)

    scatter = ax.scatter(
        line_phi["x_grid"],
        line_phi["y_grid"],
        line_phi["J"],
        c=line_phi["phi_cluster"],
        cmap="tab10",
        s=9,
        alpha=0.88,
        label="line Phi clusters",
    )
    if not intersections.empty:
        ax.scatter(intersections["x_grid"], intersections["J"], intersections["J"], color="orange", marker="X", s=90, label="equal-J intersections")

    ax.set_title("3D point cloud: x, y, J with Phi-space cluster colors")
    ax.set_xlabel("x_grid")
    ax.set_ylabel("y_grid")
    ax.set_zlabel("J(x,y)")
    ax.view_init(elev=24, azim=-61)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_j_sequence_plot(lines: pd.DataFrame, intersections: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    plt.plot(lines["x_grid"], lines["line_1_y"], label="line_1 J sequence", linewidth=2.0, color="crimson")
    plt.plot(lines["x_grid"], lines["line_2_y"], label="line_2 J sequence", linewidth=2.0, color="dodgerblue")
    plt.plot(lines["x_grid"], np.zeros(len(lines)), label="x-axis J(x,0)=0", linewidth=2.0, color="black")
    if not intersections.empty:
        plt.scatter(intersections["x_grid"], intersections["J"], s=80, marker="X", color="orange", label="same-J intersections")
    plt.title("Whole-line representation: each line as a sequence of J values")
    plt.xlabel("x_grid")
    plt.ylabel("J value")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_object_vector_distance_heatmap(distances: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(distances.to_numpy(), cmap="viridis")
    plt.colorbar(im, label="Euclidean distance in object-vector feature space")
    plt.xticks(range(len(distances.columns)), distances.columns, rotation=45, ha="right")
    plt.yticks(range(len(distances.index)), distances.index)
    plt.title("Whole-object J-vector distance matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_object_vector_mds(distances: pd.DataFrame, object_clusters: pd.DataFrame, output_path: Path) -> None:
    if len(distances) < 2:
        return
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(distances.to_numpy())
    merged = object_clusters.set_index("object_id").loc[distances.index]
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=merged["object_vector_cluster"], cmap="tab10", s=120)
    for i, object_id in enumerate(distances.index):
        plt.text(coords[i, 0] + 0.02, coords[i, 1] + 0.02, object_id, fontsize=9)
    plt.title("Whole-object J-vector clustering visualized with MDS")
    plt.xlabel("MDS dimension 1")
    plt.ylabel("MDS dimension 2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_interactive_3d(center: pd.DataFrame, line_phi: pd.DataFrame, intersections: pd.DataFrame, output_path: Path) -> None:
    if go is None:
        return
    fig = go.Figure()
    colors = {
        "line_1": "crimson",
        "line_2": "dodgerblue",
        "x_axis_J_x_0": "black",
        "y_axis_J_0_y": "gray",
    }
    axis = center[center["object_kind"] == "axis"]
    for object_id, group in axis.groupby("object_id"):
        fig.add_trace(
            go.Scatter3d(
                x=group["x_grid"], y=group["y_grid"], z=group["J"],
                mode="lines", name=object_id,
                line=dict(color=colors.get(object_id, "gray"), width=5),
            )
        )
    for cluster_id, group in line_phi.groupby("phi_cluster"):
        fig.add_trace(
            go.Scatter3d(
                x=group["x_grid"], y=group["y_grid"], z=group["J"],
                mode="markers", name=f"Phi line cluster {cluster_id}",
                marker=dict(size=3),
                text=[f"x={x}, y={y}, J={j:.2f}" for x, y, j in zip(group["x_grid"], group["y_grid"], group["J"])],
            )
        )
    if not intersections.empty:
        fig.add_trace(
            go.Scatter3d(
                x=intersections["x_grid"], y=intersections["J"], z=intersections["J"],
                mode="markers", name="equal-J intersections",
                marker=dict(size=8, color="orange", symbol="x"),
            )
        )
    fig.update_layout(
        title="Interactive 3D: x, y, J with Phi-space clusters",
        scene=dict(xaxis_title="x_grid", yaxis_title="y_grid", zaxis_title="J(x,y)"),
        width=1000,
        height=750,
    )
    fig.write_html(str(output_path))


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def run_pipeline(output_dir: Path | str = "outputs", cfg: ChartConfig | None = None) -> Dict[str, Path]:
    """Run the entire project and save data products and visualizations."""

    cfg = cfg or ChartConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate lines and intersection table.
    lines = generate_stock_like_lines(cfg)
    intersections = detect_equal_j_intersections(lines, cfg)

    # 2. Render chart and extract raster pixels from masks.
    chart_path = output_dir / "01_synthetic_stock_like_chart.png"
    masks = render_chart(lines, intersections, cfg, chart_path)
    raster_pixels = extract_raster_pixels_from_masks(masks, cfg)

    # 3. Whole-object J-vector clustering. These clusters supply the final
    # object-level lift that is added into H.
    object_vectors = build_object_j_vectors(lines, cfg)
    object_clusters, object_distances = cluster_object_j_vectors(object_vectors, cfg)

    # 4. Build the centerline feature table with Phi = [x,y,slope,curvature,J,F,H]
    # and apply the object-vector lift into H.
    center = build_centerline_feature_table(lines, cfg)
    center = apply_object_vector_lift(center, object_clusters, cfg)
    line_center = center[center["object_kind"] == "chart_line"].copy().reset_index(drop=True)

    # 5. Iterative dimensional lifting and full Phi clustering.
    lifting_dir = output_dir / "dimensional_lifting_stages"
    lifted_lines, lifting_summary = iterative_dimensional_lifting(line_center, cfg, lifting_dir)
    phi_lines = cluster_full_phi(line_center, cfg)

    # 6. Merge final labels into a clean centerline output table.
    final_center = center.copy()
    final_center["final_cluster"] = final_center["object_id"]
    final_center.loc[final_center["object_kind"].eq("chart_line"), "final_cluster"] = (
        phi_lines["phi_cluster_name"].to_numpy()
    )

    # 7. Save CSV and JSON artifacts.
    lines.to_csv(output_dir / "generated_line_sequences.csv", index=False)
    intersections.to_csv(output_dir / "equal_J_intersections.csv", index=False)
    raster_pixels.to_csv(output_dir / "raster_pixel_table.csv", index=False)
    center.to_csv(output_dir / "centerline_phi_feature_table.csv", index=False)
    phi_lines.to_csv(output_dir / "line_points_clustered_in_full_phi.csv", index=False)
    lifted_lines.to_csv(output_dir / "line_points_iterative_lifting_clusters.csv", index=False)
    lifting_summary.to_csv(output_dir / "dimensional_lifting_summary.csv", index=False)
    object_vectors.to_csv(output_dir / "whole_object_J_vectors.csv", index=False)
    object_clusters.to_csv(output_dir / "whole_object_vector_clusters.csv", index=False)
    object_distances.to_csv(output_dir / "whole_object_vector_distance_matrix.csv")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 8. Visualizations.
    save_original_chart_preview(chart_path, output_dir / "02_chart_preview.png")
    save_centerline_xy_plot(center, intersections, output_dir / "03_centerlines_axes_and_intersections_2d.png")
    save_phi_cluster_projection(phi_lines, intersections, output_dir / "04_full_phi_clusters_projected_to_2d.png")
    save_axis_and_line_clusters_2d(center, phi_lines, intersections, output_dir / "05_axes_and_phi_line_clusters_2d.png")
    save_3d_phi_plot(center, phi_lines, intersections, output_dir / "06_3d_x_y_J_phi_clusters.png")
    save_j_sequence_plot(lines, intersections, output_dir / "07_whole_line_J_sequences.png")
    save_object_vector_distance_heatmap(object_distances, output_dir / "08_whole_object_vector_distance_heatmap.png")
    save_object_vector_mds(object_distances, object_clusters, output_dir / "09_whole_object_vector_mds_clusters.png")
    save_interactive_3d(center, phi_lines, intersections, output_dir / "10_interactive_3d_x_y_J_phi_clusters.html")

    return {
        "output_dir": output_dir,
        "chart": chart_path,
        "phi_features_csv": output_dir / "centerline_phi_feature_table.csv",
        "intersections_csv": output_dir / "equal_J_intersections.csv",
        "interactive_3d": output_dir / "10_interactive_3d_x_y_J_phi_clusters.html",
    }


if __name__ == "__main__":
    run_pipeline()
