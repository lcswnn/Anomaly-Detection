"""
quality_checks.py — Pre-Ship Quality Checks for Anomaly Detection
==================================================================
Three checks to run before putting the system in users' hands:

1. Anomaly Sub-Classification
   Classifies anomalies into categories using morphological heuristics:
   - saturated_star:  bright saturated stars / diffraction spikes
   - satellite_trail: linear streaks (satellite trails, cosmic rays)
   - extended_source: diffuse extended objects (galaxy mergers, tidal tails)
   - unknown:         doesn't match known patterns — truly novel

2. Threshold Robustness Check
   Tests whether the percentile threshold is stable across data subsets
   using bootstrap resampling and field-level analysis.

3. False Negative Mining
   Finds structurally interesting tiles that the model reconstructs well
   (low MSE) but may contain scientifically interesting objects like
   galaxy mergers, lensing arcs, or interacting systems.

Usage:
    python quality_checks.py

Outputs:
    ../models/subclassifications.npy    — per-anomaly class labels
    ../models/subclass_mosaic.png       — anomaly mosaic colored by class
    ../models/threshold_robustness.png  — bootstrap threshold distribution
    ../models/false_negatives.png       — candidate false negatives
    ../models/quality_report.txt        — text summary of all checks
"""

import os
import numpy as np
import torch
import h5py
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Autoencoder

# ── Configuration ──────────────────────────────────────────────
H5_PATH = "../data/tiles/tiles.h5"
MODEL_PATH = "../models/best_model.pth"
OUTPUT_DIR = "../models"

BATCH_SIZE = 128
ANOMALY_PERCENTILE = 97

# Sub-classification thresholds (tuned for SDSS 128x128 tiles)
SATURATION_PEAK_THRESHOLD = 0.95     # peak pixel value for saturated star
SATURATION_CONCENTRATION = 0.005     # fraction of pixels above 0.8 (SDSS stars are compact)
TRAIL_LINEARITY_THRESHOLD = 3.0      # elongation ratio for linear features
TRAIL_BRIGHT_FRACTION = 0.02         # min fraction of bright pixels for trails
EXTENDED_SPREAD_THRESHOLD = 0.08     # spatial spread for extended sources

# Bootstrap config
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95  # confidence interval percentage

# False negative config
FALSE_NEG_TOP_N = 64                 # how many candidates to surface
# ───────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_reconstruction_errors(model, h5_path, device, batch_size):
    """Compute per-tile MSE. Returns (errors, reconstructions_cache_indices)."""
    model.eval()
    errors = []

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]
        total = tiles.shape[0]

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = tiles[start:end]
                x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
                recon = model(x)
                mse = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
                errors.extend(mse.cpu().numpy())

    return np.array(errors)


# ═══════════════════════════════════════════════════════════════
# 1. ANOMALY SUB-CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classify_tile(tile, error_map):
    """
    Classify a single anomalous tile into a sub-category based on
    morphological features of the tile and its error map.

    Returns one of: 'saturated_star', 'satellite_trail', 'extended_source', 'unknown'
    """
    # --- Feature extraction ---

    # Peak intensity and concentration
    peak_val = tile.max()
    bright_mask = tile > 0.8
    bright_fraction = bright_mask.mean()

    # Spatial moments of the error map for shape analysis
    # Threshold error map to focus on significant errors
    err_thresh = error_map > np.percentile(error_map, 90)

    if err_thresh.sum() < 5:
        return "unknown"

    # Connected component analysis on error map
    labeled, n_components = ndimage.label(err_thresh)

    # Get the largest connected component
    if n_components > 0:
        component_sizes = ndimage.sum(err_thresh, labeled, range(1, n_components + 1))
        largest_label = np.argmax(component_sizes) + 1
        largest_component = labeled == largest_label
    else:
        largest_component = err_thresh

    # Compute inertia tensor for shape (elongation)
    coords = np.argwhere(largest_component)
    if len(coords) < 3:
        return "unknown"

    cov = np.cov(coords.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Elongation ratio: ratio of major to minor axis
    elongation = eigenvalues[0] / max(eigenvalues[1], 1e-8)

    # Spatial spread: how much of the tile area the error covers
    error_spread = err_thresh.mean()

    # Radial concentration: does the error peak at center of bright region?
    if bright_mask.sum() > 0:
        bright_coords = np.argwhere(bright_mask)
        centroid = bright_coords.mean(axis=0)
        distances = np.sqrt(((bright_coords - centroid) ** 2).sum(axis=1))
        radial_std = distances.std() if len(distances) > 1 else 0
    else:
        radial_std = 999  # no bright region

    # --- Classification rules ---

    # Saturated star / diffraction spike:
    #   High peak pixel, concentrated bright region, moderate elongation
    if (peak_val > SATURATION_PEAK_THRESHOLD
            and bright_fraction > SATURATION_CONCENTRATION):
        return "saturated_star"

    # Satellite trail / cosmic ray:
    #   Highly elongated error pattern, narrow bright streak
    if (elongation > TRAIL_LINEARITY_THRESHOLD
            and bright_fraction > TRAIL_BRIGHT_FRACTION
            and error_spread < 0.15):
        return "satellite_trail"

    # Extended source (galaxy merger, tidal features):
    #   Spread out error, moderate brightness, not highly elongated
    if (error_spread > EXTENDED_SPREAD_THRESHOLD
            and elongation < TRAIL_LINEARITY_THRESHOLD
            and peak_val < SATURATION_PEAK_THRESHOLD):
        return "extended_source"

    return "unknown"


def subclassify_anomalies(model, h5_path, errors, threshold, device):
    """
    Classify all anomalous tiles into sub-categories.

    Returns:
        anomaly_indices: array of indices for anomalous tiles
        labels: list of string labels for each anomaly
        counts: dict of {label: count}
    """
    model.eval()
    anomaly_indices = np.where(errors > threshold)[0]

    # Sort by error descending
    sorted_order = np.argsort(errors[anomaly_indices])[::-1]
    anomaly_indices = anomaly_indices[sorted_order]

    labels = []

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]

        with torch.no_grad():
            for idx in anomaly_indices:
                tile = tiles[idx]
                x = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                recon = model(x)
                reconstructed = recon.squeeze().cpu().numpy()
                error_map = (tile - reconstructed) ** 2

                label = classify_tile(tile, error_map)
                labels.append(label)

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return anomaly_indices, labels, counts


def plot_subclass_mosaic(h5_path, anomaly_indices, labels, errors, save_path, n=64):
    """Plot top anomalies in a grid, color-coded by sub-class."""
    CLASS_COLORS = {
        "saturated_star":  "#ff4444",
        "satellite_trail": "#44aaff",
        "extended_source": "#44ff44",
        "unknown":         "#ffaa00",
    }

    n = min(n, len(anomaly_indices))
    cols = 8
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]

        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            if i < n:
                idx = anomaly_indices[i]
                tile = tiles[idx]
                label = labels[i]
                color = CLASS_COLORS[label]

                ax.imshow(tile, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_title(f"{label}\nMSE:{errors[idx]:.4f}", fontsize=6,
                             color=color, fontweight="bold", pad=2)

                # Color border by class
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
            else:
                ax.axis("off")
                continue

            ax.set_xticks([])
            ax.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Top {n} Anomalies — Sub-Classified", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sub-classified mosaic saved to {save_path}")


# ═══════════════════════════════════════════════════════════════
# 2. THRESHOLD ROBUSTNESS CHECK
# ═══════════════════════════════════════════════════════════════

def check_threshold_robustness(errors, percentile=97, n_bootstrap=1000, ci=95):
    """
    Bootstrap the error distribution to assess threshold stability.

    Returns:
        threshold: original threshold
        ci_low, ci_high: confidence interval bounds
        bootstrap_thresholds: all bootstrapped thresholds
        cv: coefficient of variation
    """
    n = len(errors)
    rng = np.random.default_rng(42)

    bootstrap_thresholds = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(errors, size=n, replace=True)
        bootstrap_thresholds[i] = np.percentile(sample, percentile)

    threshold = np.percentile(errors, percentile)
    alpha = (100 - ci) / 2
    ci_low = np.percentile(bootstrap_thresholds, alpha)
    ci_high = np.percentile(bootstrap_thresholds, 100 - alpha)
    cv = bootstrap_thresholds.std() / bootstrap_thresholds.mean()

    return threshold, ci_low, ci_high, bootstrap_thresholds, cv


def check_field_stability(h5_path, errors, percentile=97):
    """
    Check threshold stability across different data chunks (simulating
    different SDSS fields). Splits tiles into 10 equal chunks and
    computes per-chunk thresholds.

    Returns:
        chunk_thresholds: array of per-chunk thresholds
        overall_threshold: threshold from full dataset
        max_deviation: max absolute deviation from overall threshold
    """
    n = len(errors)
    n_chunks = 10
    chunk_size = n // n_chunks

    chunk_thresholds = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n
        chunk_errors = errors[start:end]
        chunk_thresholds.append(np.percentile(chunk_errors, percentile))

    chunk_thresholds = np.array(chunk_thresholds)
    overall_threshold = np.percentile(errors, percentile)
    max_deviation = np.max(np.abs(chunk_thresholds - overall_threshold))

    return chunk_thresholds, overall_threshold, max_deviation


def plot_threshold_robustness(bootstrap_thresholds, threshold, ci_low, ci_high,
                              chunk_thresholds, save_path):
    """Plot bootstrap distribution and field-level stability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bootstrap distribution
    ax1.hist(bootstrap_thresholds, bins=50, color="steelblue", alpha=0.8, edgecolor="none")
    ax1.axvline(threshold, color="red", linewidth=2, linestyle="-",
                label=f"Threshold: {threshold:.6f}")
    ax1.axvline(ci_low, color="orange", linewidth=1.5, linestyle="--",
                label=f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    ax1.axvline(ci_high, color="orange", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Threshold Value", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Bootstrap Threshold Distribution\n(1000 resamples)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Per-chunk stability
    chunks = range(1, len(chunk_thresholds) + 1)
    ax2.bar(chunks, chunk_thresholds, color="steelblue", alpha=0.8)
    ax2.axhline(threshold, color="red", linewidth=2, linestyle="-",
                label=f"Overall: {threshold:.6f}")
    ax2.set_xlabel("Data Chunk (simulated field)", fontsize=11)
    ax2.set_ylabel("Threshold Value", fontsize=11)
    ax2.set_title("Threshold Stability Across Data Chunks", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Threshold robustness plot saved to {save_path}")


# ═══════════════════════════════════════════════════════════════
# 3. FALSE NEGATIVE MINING
# ═══════════════════════════════════════════════════════════════

def compute_structural_interest(tile):
    """
    Score a tile's structural complexity / scientific interest.
    Combines multiple morphological features that correlate with
    interesting astronomical objects (mergers, lensing, tidal features).

    Returns a float score (higher = more structurally interesting).
    """
    # Gradient magnitude — measures edge/structure content
    gy, gx = np.gradient(tile)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    gradient_score = grad_mag.mean()

    # Number of distinct bright regions (compact sources)
    bright_mask = tile > 0.5
    labeled, n_sources = ndimage.label(bright_mask)

    # Filter out tiny noise blobs — keep only regions > 20 pixels
    significant_sources = 0
    if n_sources > 0:
        for lbl in range(1, n_sources + 1):
            if np.sum(labeled == lbl) > 20:
                significant_sources += 1

    # Asymmetry — interesting objects are often asymmetric
    # Compare tile to its 180-degree rotation
    rotated = np.rot90(tile, 2)
    asymmetry = np.mean(np.abs(tile - rotated))

    # Multi-scale structure: compare local vs global contrast
    # High ratio means localized features (galaxies, arcs) vs smooth background
    local_std = ndimage.uniform_filter(tile ** 2, size=16) - \
                ndimage.uniform_filter(tile, size=16) ** 2
    local_std = np.sqrt(np.clip(local_std, 0, None))
    structure_ratio = local_std.mean() / max(tile.std(), 1e-8)

    # Weighted combination
    score = (
        gradient_score * 2.0 +
        significant_sources * 0.3 +
        asymmetry * 3.0 +
        structure_ratio * 1.5
    )

    return score


def mine_false_negatives(model, h5_path, errors, threshold, device,
                         top_n=64, batch_size=128):
    """
    Find tiles below the anomaly threshold that are structurally interesting.
    These are potential false negatives — scientifically interesting objects
    that the autoencoder reconstructs well.

    Strategy:
        1. Filter to tiles BELOW threshold (classified as "normal")
        2. Exclude very low-error tiles (truly boring background)
        3. Score remaining tiles by structural interest
        4. Return top N most interesting "normal" tiles

    Returns:
        candidate_indices: indices of candidate false negatives
        interest_scores: structural interest score per candidate
    """
    # Focus on tiles in the 50th-97th percentile range:
    # below threshold but not dead-boring background
    low_bound = np.percentile(errors, 50)
    normal_mask = (errors <= threshold) & (errors >= low_bound)
    normal_indices = np.where(normal_mask)[0]

    print(f"  Scanning {len(normal_indices)} normal tiles for false negatives...")

    interest_scores = np.zeros(len(normal_indices))

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]

        for i, idx in enumerate(normal_indices):
            tile = tiles[idx]
            interest_scores[i] = compute_structural_interest(tile)

            if (i + 1) % 5000 == 0:
                print(f"    Scored {i+1}/{len(normal_indices)} tiles")

    # Take the top N most interesting
    top_order = np.argsort(interest_scores)[::-1][:top_n]
    candidate_indices = normal_indices[top_order]
    candidate_scores = interest_scores[top_order]

    return candidate_indices, candidate_scores


def plot_false_negatives(h5_path, candidate_indices, candidate_scores,
                         errors, save_path, n=64):
    """Plot candidate false negatives ranked by structural interest."""
    n = min(n, len(candidate_indices))
    cols = 8
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]

        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            if i < n:
                idx = candidate_indices[i]
                tile = tiles[idx]
                ax.imshow(tile, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_title(f"Interest:{candidate_scores[i]:.3f}\n"
                             f"MSE:{errors[idx]:.5f}",
                             fontsize=6, pad=2)
            else:
                ax.axis("off")
                continue

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        f"Top {n} Candidate False Negatives\n"
        "(Structurally interesting tiles classified as normal)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  False negative candidates saved to {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN — Run All Quality Checks
# ═══════════════════════════════════════════════════════════════

def run_all_checks(model=None, errors=None, threshold=None, device=None):
    """
    Run all three quality checks. Can be called standalone or from evaluate.py.

    Args:
        model: loaded Autoencoder (loaded from disk if None)
        errors: precomputed reconstruction errors (recomputed if None)
        threshold: anomaly threshold (recomputed if None)
        device: torch device (auto-detected if None)
    """
    if device is None:
        device = get_device()

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("QUALITY CHECK REPORT")
    report_lines.append("=" * 60)

    # Load model if needed
    if model is None:
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()

    # Compute errors if needed
    if errors is None:
        print("\n  Computing reconstruction errors...")
        errors = compute_reconstruction_errors(model, H5_PATH, device, BATCH_SIZE)

    # Compute threshold if needed
    if threshold is None:
        threshold = np.percentile(errors, ANOMALY_PERCENTILE)

    n_anomalies = np.sum(errors > threshold)

    # ── Check 1: Anomaly Sub-Classification ───────────────────
    print("\n── Check 1: Anomaly Sub-Classification ──────────────")
    anomaly_indices, labels, counts = subclassify_anomalies(
        model, H5_PATH, errors, threshold, device
    )

    print(f"  Total anomalies: {len(anomaly_indices)}")
    report_lines.append(f"\n1. ANOMALY SUB-CLASSIFICATION")
    report_lines.append(f"   Total anomalies: {len(anomaly_indices)}")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(anomaly_indices)
        print(f"    {cls:20s}: {count:5d} ({pct:5.1f}%)")
        report_lines.append(f"   {cls:20s}: {count:5d} ({pct:5.1f}%)")

    # Flag if one class dominates
    dominant_cls = max(counts, key=counts.get)
    dominant_pct = 100 * counts[dominant_cls] / len(anomaly_indices)
    if dominant_pct > 60:
        warning = (f"\n   WARNING: '{dominant_cls}' dominates at {dominant_pct:.1f}%.\n"
                   f"   Consider filtering these out or adjusting detection to focus on rarer types.")
        print(f"  {warning}")
        report_lines.append(warning)

    # Save labels
    np.save(os.path.join(OUTPUT_DIR, "subclassifications.npy"),
            {"indices": anomaly_indices, "labels": labels, "counts": counts},
            allow_pickle=True)

    plot_subclass_mosaic(
        H5_PATH, anomaly_indices, labels, errors,
        os.path.join(OUTPUT_DIR, "subclass_mosaic.png")
    )

    # ── Check 2: Threshold Robustness ─────────────────────────
    print("\n── Check 2: Threshold Robustness ────────────────────")
    thresh, ci_low, ci_high, bootstrap_thresholds, cv = check_threshold_robustness(
        errors, ANOMALY_PERCENTILE, N_BOOTSTRAP, BOOTSTRAP_CI
    )
    chunk_thresholds, overall_thresh, max_dev = check_field_stability(
        H5_PATH, errors, ANOMALY_PERCENTILE
    )

    # Count range of anomalies under CI bounds
    n_at_low = np.sum(errors > ci_low)
    n_at_high = np.sum(errors > ci_high)

    print(f"  Threshold: {thresh:.6f}")
    print(f"  95% CI:    [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  CV:        {cv:.4f} ({cv*100:.2f}%)")
    print(f"  Anomalies at CI bounds: {n_at_high} — {n_at_low}")
    print(f"  Per-chunk max deviation: {max_dev:.6f} ({100*max_dev/thresh:.1f}% of threshold)")

    report_lines.append(f"\n2. THRESHOLD ROBUSTNESS")
    report_lines.append(f"   Threshold: {thresh:.6f}")
    report_lines.append(f"   95% Bootstrap CI: [{ci_low:.6f}, {ci_high:.6f}]")
    report_lines.append(f"   Coefficient of Variation: {cv:.4f} ({cv*100:.2f}%)")
    report_lines.append(f"   Anomaly count range at CI bounds: {n_at_high} — {n_at_low}")
    report_lines.append(f"   Per-chunk max deviation: {max_dev:.6f} ({100*max_dev/thresh:.1f}%)")

    if cv > 0.05:
        warning = (f"\n   WARNING: High threshold variability (CV={cv:.2f}).\n"
                   f"   The threshold is sensitive to the training set composition.\n"
                   f"   Consider using a larger or more diverse training set.")
        print(f"  {warning}")
        report_lines.append(warning)
    else:
        msg = "\n   PASS: Threshold is stable across resamples."
        print(f"  {msg}")
        report_lines.append(msg)

    if max_dev / thresh > 0.20:
        warning = (f"\n   WARNING: Large per-chunk deviation ({100*max_dev/thresh:.1f}%).\n"
                   f"   Different sky regions produce different thresholds.\n"
                   f"   Adding new SDSS fields may require recalibration.")
        print(f"  {warning}")
        report_lines.append(warning)
    else:
        msg = "\n   PASS: Threshold is consistent across data chunks."
        print(f"  {msg}")
        report_lines.append(msg)

    plot_threshold_robustness(
        bootstrap_thresholds, thresh, ci_low, ci_high, chunk_thresholds,
        os.path.join(OUTPUT_DIR, "threshold_robustness.png")
    )

    # ── Check 3: False Negative Mining ────────────────────────
    print("\n── Check 3: False Negative Mining ───────────────────")
    candidate_indices, candidate_scores = mine_false_negatives(
        model, H5_PATH, errors, threshold, device,
        top_n=FALSE_NEG_TOP_N, batch_size=BATCH_SIZE
    )

    print(f"  Surfaced {len(candidate_indices)} candidate false negatives")
    print(f"  Interest score range: {candidate_scores.min():.3f} — {candidate_scores.max():.3f}")

    report_lines.append(f"\n3. FALSE NEGATIVE MINING")
    report_lines.append(f"   Candidates surfaced: {len(candidate_indices)}")
    report_lines.append(f"   Interest score range: {candidate_scores.min():.3f} — {candidate_scores.max():.3f}")
    report_lines.append(f"   These are tiles classified as NORMAL but with high structural complexity.")
    report_lines.append(f"   Manually inspect ../models/false_negatives.png for galaxy mergers,")
    report_lines.append(f"   lensing arcs, or other scientifically interesting objects.")

    plot_false_negatives(
        H5_PATH, candidate_indices, candidate_scores, errors,
        os.path.join(OUTPUT_DIR, "false_negatives.png")
    )

    # ── Save Report ───────────────────────────────────────────
    report_lines.append(f"\n{'=' * 60}")
    report_text = "\n".join(report_lines)

    report_path = os.path.join(OUTPUT_DIR, "quality_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Quality report saved to {report_path}")

    return {
        "anomaly_indices": anomaly_indices,
        "labels": labels,
        "counts": counts,
        "threshold": thresh,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cv": cv,
        "chunk_thresholds": chunk_thresholds,
        "false_neg_indices": candidate_indices,
        "false_neg_scores": candidate_scores,
    }


if __name__ == "__main__":
    print("Quality Checks — Pre-Ship Validation")
    print("=" * 40)
    results = run_all_checks()
    print("\nAll quality checks complete!")
