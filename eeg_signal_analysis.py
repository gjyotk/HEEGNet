#!/usr/bin/env python3
"""
Comprehensive EEG Signal Analysis: ERPs, Spectrograms, Time-Frequency, & SNR
=============================================================================
THINGS-EEG2 Dataset (Test Partition: 200 concepts × 80 reps × 63 scalp channels)

This script conducts an in-depth neurophysiological signal analysis:
1. Evoked Response Potentials (ERP):
   - Grand-average Butterfly plot across all 63 electrodes (-200 to +790 ms).
   - Global Field Power (GFP) tracing temporal dynamics and canonical visual peak latencies.
   - Region-of-Interest (ROI) ERP waveforms (Occipital O1/Oz/O2, Parieto-Occipital PO7/PO8,
     Centroparietal Cz/Pz, Frontal Fz/AFz) with 95% confidence bands.
   - Cross-subject ERP comparison explicitly contrasting high-SNR (Sub-08, Sub-07, Sub-03)
     vs. low-SNR (Sub-09, Sub-02) subjects to reveal the biological basis of RDM reliability.
2. Scalp Topographical Potential Mapping:
   - 2D interpolated scalp topographies at canonical post-stimulus latencies:
     60 ms (early sensory), 110 ms (P100), 170 ms (N170), 230 ms (P200), 350 ms (P300), 550 ms (Late).
3. Time-Frequency Analysis & Spectrograms:
   - Complex Morlet Wavelet Decomposition across 4–45 Hz.
   - Event-Related Spectral Perturbation (ERSP) in decibels (dB) relative to baseline (-200 to 0 ms).
   - Inter-Trial Phase Coherence (ITPC / Phase-Locking Value PLV) quantifying phase resetting.
4. Power Spectral Density (PSD) & Alpha Desynchronization (ERD/ERS):
   - Welch's PSD comparing pre-stimulus baseline vs. post-stimulus evoked/induced windows.
   - Quantification of event-related desynchronization (ERD%) across standard frequency bands
     (Delta, Theta, Alpha, Beta, Low Gamma).
5. Evoked vs. Induced Power Decomposition & SNR across all 10 subjects.

Deliverables:
  - High-resolution figures in results/signal_analysis/figures/
  - Quantitative metrics tables in results/signal_analysis/tables/
  - Integrated Executive Signal Analysis Dashboard.

Usage:
    /tmp/cv13_climb/.venv/bin/python eeg_signal_analysis.py
"""

import os
import sys
import time
import zipfile
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.interpolate import griddata

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Polygon

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Directories & Constants
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "eeg-dataset", "derivatives", "preprocessed_eeg")
COORD_FILE = os.path.join(BASE_DIR, "eeg-dataset", "sub-09", "ses-01", "eeg", "sub-09_ses-01_space-CapTrak_electrodes.tsv")
OUT_DIR = os.path.join(BASE_DIR, "results", "signal_analysis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

SR = 100  # Sampling rate: 100 Hz
TIMES = np.arange(-20, 80) / SR  # -0.20 to +0.79 s (100 samples)
BASELINE_SLICE = slice(0, 20)     # -200 ms to 0 ms (20 samples)
POSTSTIM_SLICE = slice(20, 100)   # 0 ms to 790 ms (80 samples)
ALL_SUBJECTS = list(range(1, 11))


# =============================================================================
# Helper: Load Scalp Coordinates
# =============================================================================
def load_electrode_coords():
    """Load 3D electrode coordinates and project to 2D normalized circle."""
    df_coord = pd.read_csv(COORD_FILE, sep="\t")
    # x: left-to-right (lateral), y: posterior-to-anterior, z: vertical
    x_3d = df_coord["x"].values
    y_3d = df_coord["y"].values
    z_3d = df_coord["z"].values
    ch_names = list(df_coord["name"].values)

    # Azimuthal equidistant projection onto unit disc
    r_3d = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
    # Polar angles
    theta = np.arctan2(y_3d, x_3d)
    phi = np.arccos(np.clip(z_3d / (r_3d + 1e-12), -1.0, 1.0))  # zenith angle from top of head
    # Normalized radial distance on 2D scalp
    r_2d = phi / (np.pi / 2.0) * 0.45  # scale to disc of radius 0.45
    x_2d = r_2d * np.cos(theta)
    y_2d = r_2d * np.sin(theta)

    return ch_names, x_2d, y_2d


# =============================================================================
# Helper: Data Loading
# =============================================================================
def load_subject_data(sub_id: int):
    """Load test EEG array: (200 concepts, 80 reps, 63 channels, 100 times)."""
    zf_name = f"sub-{sub_id:02d}__63_channels.zip"
    zf_path = os.path.join(PREPROCESSED_DIR, zf_name)
    npy_name = f"sub-{sub_id:02d}__63_channels/preprocessed_eeg_test.npy"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open(npy_name) as fp:
            raw = np.load(fp, allow_pickle=True).item()
    data = raw["preprocessed_eeg_data"][:, :, :63, :]  # drop channel 64 ('stim')
    ch_names = list(raw["ch_names"][:63])
    return data, ch_names


# =============================================================================
# Helper: Complex Morlet Wavelet Transform
# =============================================================================
def morlet_cwt(data, sr=100, freqs=np.linspace(4, 45, 35), n_cycles=5):
    """
    Continuous wavelet transform via FFT convolution with complex Morlet wavelets.
    data: (..., n_times)
    Returns: (..., n_freqs, n_times) complex coefficients.
    """
    n_times = data.shape[-1]
    n_freqs = len(freqs)
    out_shape = data.shape[:-1] + (n_freqs, n_times)

    # Time support for wavelet (1.0 sec window centered at 0)
    t_wav = np.arange(-int(sr * 0.5), int(sr * 0.5) + 1) / sr
    n_conv = n_times + len(t_wav) - 1

    # Frequency domain of data
    data_fft = np.fft.fft(data, n=n_conv, axis=-1)
    cwt = np.zeros(out_shape, dtype=np.complex128)

    for i, f in enumerate(freqs):
        s = n_cycles / (2.0 * np.pi * f)
        wavelet = np.exp(-t_wav**2 / (2.0 * s**2)) * np.exp(1j * 2.0 * np.pi * f * t_wav)
        # Normalization factor for unit energy
        wavelet /= np.sqrt(0.5) * (s * np.sqrt(np.pi))**0.5
        w_fft = np.fft.fft(wavelet, n=n_conv)
        conv = np.fft.ifft(data_fft * w_fft, axis=-1)
        half_w = len(t_wav) // 2
        cwt[..., i, :] = conv[..., half_w:half_w + n_times]

    return cwt


# =============================================================================
# Scalp Topography Plotter
# =============================================================================
def plot_scalp_topomap(ax, values, x_coords, y_coords, vmin=None, vmax=None, cmap="RdBu_r", title=None):
    """Plot an interpolated 2D scalp topography with head boundary and nose."""
    if vmin is None or vmax is None:
        abs_max = np.percentile(np.abs(values), 98)
        vmin, vmax = -abs_max, abs_max

    grid_x, grid_y = np.mgrid[-0.5:0.5:100j, -0.5:0.5:100j]
    grid_z = griddata((x_coords, y_coords), values, (grid_x, grid_y), method="cubic", fill_value=np.nan)

    # Mask outside unit head radius (radius 0.45)
    mask = (grid_x**2 + grid_y**2) > (0.45**2)
    grid_z[mask] = np.nan

    im = ax.imshow(grid_z.T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap=cmap, vmin=vmin, vmax=vmax)

    # Head outline
    head = Circle((0, 0), 0.45, fill=False, color="black", linewidth=1.5)
    ax.add_patch(head)

    # Nose
    nose = Polygon([(-0.06, 0.45), (0.0, 0.51), (0.06, 0.45)], fill=False, color="black", linewidth=1.5)
    ax.add_patch(nose)

    # Ears
    l_ear = Circle((-0.46, 0.0), 0.04, fill=False, color="black", linewidth=1.2)
    r_ear = Circle((0.46, 0.0), 0.04, fill=False, color="black", linewidth=1.2)
    ax.add_patch(l_ear)
    ax.add_patch(r_ear)

    # Electrodes
    ax.scatter(x_coords, y_coords, c="black", s=6, alpha=0.7)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    return im


# =============================================================================
# Main Analysis Pipeline
# =============================================================================
def run_eeg_signal_analysis():
    print("=" * 80)
    print("COMPREHENSIVE EEG SIGNAL ANALYSIS: ERPs, SPECTROGRAMS, TOPOMAPS, & SNR")
    print("=" * 80)
    t_start = time.time()

    ch_names_ref, x_coords, y_coords = load_electrode_coords()
    ch_dict = {name: i for i, name in enumerate(ch_names_ref)}

    # Define key ROIs
    parieto_occ_channels = [ch_dict[c] for c in ["PO3", "PO4", "PO7", "PO8", "O1", "O2"] if c in ch_dict]
    roi_indices = {
        "Occipital (O1, Oz, O2)": [ch_dict[c] for c in ["O1", "Oz", "O2"] if c in ch_dict],
        "Lateral Parieto-Occipital (PO7, PO8)": [ch_dict[c] for c in ["PO7", "PO8"] if c in ch_dict],
        "Centroparietal (Cz, Pz, CPz)": [ch_dict[c] for c in ["Cz", "Pz", "CPz"] if c in ch_dict],
        "Frontal (Fz, AFz)": [ch_dict[c] for c in ["AFz", "Fz"] if c in ch_dict or c == "AFz" and "AFz" in ch_dict],
    }

    # Storage across subjects
    all_erps = []           # (10, 63, 100) - subject grand average ERPs
    all_gfps = []           # (10, 100)     - GFP per subject
    subject_peak_metrics = []
    subject_snr_metrics = []

    print("\n[Phase 1] Ingesting EEG recordings & computing ERPs across all 10 subjects...")
    for sub in ALL_SUBJECTS:
        data, ch_names = load_subject_data(sub)
        # data shape: (200 concepts, 80 reps, 63 ch, 100 time)
        # Average across all 200 concepts and 80 reps -> (63, 100) ERP
        erp = np.mean(data, axis=(0, 1))
        # Baseline correction (-200 to 0 ms)
        baseline = erp[:, BASELINE_SLICE].mean(axis=-1, keepdims=True)
        erp = erp - baseline
        all_erps.append(erp)

        # Global Field Power (GFP): spatial standard deviation across channels
        gfp = np.std(erp, axis=0)
        all_gfps.append(gfp)

        # Detect P100:
        # Instead of naively searching midline Oz alone (which is near the dipole transition zone in this montage),
        # identify the early visual sensory latency from Global Field Power (GFP) in the canonical 80–130 ms window,
        # then report both the positive peak across parieto-occipital channels and the midline Oz value.
        p100_win = slice(28, 34)  # 80 to 130 ms
        p100_idx = 28 + np.argmax(gfp[p100_win])
        p100_lat = TIMES[p100_idx] * 1000.0

        p100_po_amps = erp[parieto_occ_channels, p100_idx]
        best_po_idx = np.argmax(p100_po_amps)
        p100_amp_po = float(p100_po_amps[best_po_idx])
        p100_po_ch = ch_names_ref[parieto_occ_channels[best_po_idx]]
        p100_oz_amp = float(erp[ch_dict["Oz"], p100_idx])

        # Detect N170 at PO7/PO8 average (latency in 130–220 ms)
        po_idx = roi_indices["Lateral Parieto-Occipital (PO7, PO8)"]
        po_erp = erp[po_idx, :].mean(axis=0)
        n170_win = slice(33, 43)  # 130 to 230 ms
        n170_idx = 33 + np.argmin(po_erp[n170_win])
        n170_lat = TIMES[n170_idx] * 1000.0
        n170_amp = po_erp[n170_idx]

        # Detect P300 at Pz (latency in 250–450 ms)
        pz_idx = ch_dict["Pz"]
        pz_erp = erp[pz_idx, :]
        p300_win = slice(45, 65)  # 250 to 450 ms
        p300_idx = 45 + np.argmax(pz_erp[p300_win])
        p300_lat = TIMES[p300_idx] * 1000.0
        p300_amp = pz_erp[p300_idx]

        # Evoked vs Trial Residual Noise Variance
        # Residual variance across trials around concept means
        concept_means = data.mean(axis=1)  # (200, 63, 100)
        residuals = data - concept_means[:, None, :, :]  # (200, 80, 63, 100)
        trial_var = float(np.var(residuals[:, :, :, POSTSTIM_SLICE]))

        # Common stimulus-evoked potential power (grand-average ERP across all concepts)
        evoked_power = float(np.var(erp[:, POSTSTIM_SLICE]))
        evoked_snr = evoked_power / (trial_var + 1e-12)

        # Concept-discriminative signal variance (variance across concept ERPs around grand ERP)
        concept_signal_var = float(np.var(concept_means[:, :, POSTSTIM_SLICE] - erp[None, :, POSTSTIM_SLICE]))
        concept_snr = concept_signal_var / (trial_var + 1e-12)

        subject_peak_metrics.append({
            "subject": sub,
            "P100_lat_ms": round(p100_lat, 1),
            "P100_amp_uv": round(p100_amp_po, 3),
            "P100_max_channel": p100_po_ch,
            "P100_Oz_amp_uv": round(p100_oz_amp, 3),
            "N170_lat_ms": round(n170_lat, 1),
            "N170_amp_uv": round(float(n170_amp), 3),
            "P300_lat_ms": round(p300_lat, 1),
            "P300_amp_uv": round(float(p300_amp), 3),
            "max_gfp_uv": round(float(np.max(gfp[POSTSTIM_SLICE])), 3),
            "max_gfp_lat_ms": round(float(TIMES[POSTSTIM_SLICE][np.argmax(gfp[POSTSTIM_SLICE])] * 1000), 1),
        })

        subject_snr_metrics.append({
            "subject": sub,
            "evoked_power": round(evoked_power, 6),
            "concept_signal_var": round(concept_signal_var, 6),
            "trial_noise_var": round(trial_var, 4),
            "evoked_snr_ratio": round(evoked_snr, 6),
            "evoked_snr_x1000": round(evoked_snr * 1000, 2),
            "concept_snr_ratio": round(concept_snr, 6),
            "concept_snr_x1000": round(concept_snr * 1000, 2),
        })

    all_erps = np.array(all_erps)  # (10, 63, 100)
    all_gfps = np.array(all_gfps)  # (10, 100)
    grand_erp = np.mean(all_erps, axis=0)  # (63, 100)
    grand_gfp = np.mean(all_gfps, axis=0)

    df_peaks = pd.DataFrame(subject_peak_metrics)
    df_snr = pd.DataFrame(subject_snr_metrics)
    df_peaks.to_csv(os.path.join(TAB_DIR, "subject_erp_peak_metrics.csv"), index=False)
    df_snr.to_csv(os.path.join(TAB_DIR, "subject_snr_evoked_induced_summary.csv"), index=False)

    print("\n--- Summary of ERP Peaks & Evoked SNR Across Subjects ---")
    print(df_peaks.to_string(index=False))
    print("\n--- Evoked SNR Ratios ---")
    print(df_snr.to_string(index=False))

    # =========================================================================
    # Figure 1: Butterfly Plot & Global Field Power (GFP)
    # =========================================================================
    print("\n[Phase 2] Generating Figure 1: Butterfly Plot & Global Field Power...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    times_ms = TIMES * 1000.0

    # Butterfly plot: all 63 channels
    for ch_i in range(63):
        color = "crimson" if ch_i in roi_indices["Occipital (O1, Oz, O2)"] else \
                "royalblue" if ch_i in roi_indices["Lateral Parieto-Occipital (PO7, PO8)"] else "gray"
        alpha = 0.85 if color != "gray" else 0.35
        lw = 1.6 if color != "gray" else 0.8
        axes[0].plot(times_ms, grand_erp[ch_i], color=color, alpha=alpha, linewidth=lw)

    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2, label="Stimulus Onset (0 ms)")
    axes[0].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Amplitude (μV)", fontsize=11, fontweight="bold")
    axes[0].set_title("Grand-Average ERP Butterfly Plot (63 Channels, N=10 Subjects, 160,000 Trials)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Custom legend for butterfly plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="crimson", lw=2, label="Occipital (O1, Oz, O2)"),
        Line2D([0], [0], color="royalblue", lw=2, label="Lateral Parieto-Occipital (PO7, PO8)"),
        Line2D([0], [0], color="gray", lw=1, alpha=0.6, label="Other Scalp Channels"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    # Global Field Power (GFP)
    axes[1].plot(times_ms, grand_gfp, color="darkmagenta", linewidth=2.5, label="Global Field Power (GFP)")
    axes[1].fill_between(times_ms, 0, grand_gfp, color="darkmagenta", alpha=0.2)

    # Annotate canonical GFP peaks
    gfp_p100_idx = 27 + np.argmax(grand_gfp[27:35])
    gfp_n170_idx = 35 + np.argmax(grand_gfp[35:43])
    axes[1].scatter([times_ms[gfp_p100_idx], times_ms[gfp_n170_idx]],
                    [grand_gfp[gfp_p100_idx], grand_gfp[gfp_n170_idx]],
                    color="darkmagenta", s=50, zorder=5)
    axes[1].text(times_ms[gfp_p100_idx], grand_gfp[gfp_p100_idx] + 0.05, f"P100 Peak ({times_ms[gfp_p100_idx]:.0f} ms)",
                 ha="center", fontsize=9, fontweight="bold", color="darkmagenta")
    axes[1].text(times_ms[gfp_n170_idx], grand_gfp[gfp_n170_idx] + 0.05, f"N170 Peak ({times_ms[gfp_n170_idx]:.0f} ms)",
                 ha="center", fontsize=9, fontweight="bold", color="darkmagenta")

    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Time Relative to Stimulus Onset (ms)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("GFP (μV)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, np.max(grand_gfp) * 1.25)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_erp_butterfly_and_gfp.png"), dpi=300)
    plt.close()
    print("  -> Saved: fig1_erp_butterfly_and_gfp.png")

    # =========================================================================
    # Figure 2: Region-of-Interest (ROI) ERP Waveforms
    # =========================================================================
    print("\n[Phase 3] Generating Figure 2: ROI Waveforms with Confidence Intervals...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    roi_colors = ["crimson", "royalblue", "darkgreen", "darkorange"]

    for idx, (roi_name, ch_list) in enumerate(roi_indices.items()):
        ax = axes[idx]
        color = roi_colors[idx]

        # Extract per-subject ROI waveform: (10, 100)
        sub_roi_waves = all_erps[:, ch_list, :].mean(axis=1)
        mean_wave = sub_roi_waves.mean(axis=0)
        sem_wave = stats.sem(sub_roi_waves, axis=0)

        # Plot 95% CI
        ci_95 = 1.96 * sem_wave
        ax.plot(times_ms, mean_wave, color=color, linewidth=2.5, label=f"Mean ERP ({len(ch_list)} ch)")
        ax.fill_between(times_ms, mean_wave - ci_95, mean_wave + ci_95, color=color, alpha=0.25, label="95% CI across Subjects")

        # Plot individual subject traces faintly
        for sub_i in range(10):
            ax.plot(times_ms, sub_roi_waves[sub_i], color=color, alpha=0.2, linewidth=0.8)

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(roi_name, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        if idx >= 2:
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
        if idx % 2 == 0:
            ax.set_ylabel("Amplitude (μV)", fontsize=11, fontweight="bold")

    plt.suptitle("Grand-Average Regional ERP Waveforms (N=10 Subjects, Mean ± 95% CI)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_erp_roi_waveforms.png"), dpi=300)
    plt.close()
    print("  -> Saved: fig2_erp_roi_waveforms.png")

    # =========================================================================
    # Figure 3: Scalp Topographies at Canonical Latencies
    # =========================================================================
    print("\n[Phase 4] Generating Figure 3: 2D Scalp Topographies across Canonical Latencies...")
    target_latencies_ms = [60, 110, 170, 230, 350, 550]
    fig, axes = plt.subplots(1, len(target_latencies_ms), figsize=(18, 3.8), dpi=300)

    # Determine global symmetrical colorbar limits across canonical maps
    sample_indices = [int(round((lat / 1000.0 - (-0.2)) * SR)) for lat in target_latencies_ms]
    map_vals = [grand_erp[:, s_idx] for s_idx in sample_indices]
    v_max = np.percentile(np.abs(map_vals), 98)
    v_min = -v_max

    for i, (lat, s_idx) in enumerate(zip(target_latencies_ms, sample_indices)):
        vals = grand_erp[:, s_idx]
        title_tag = "P100 (Early Visual)" if lat == 110 else \
                    "N170 (Category)" if lat == 170 else \
                    "P200 (Visual Div.)" if lat == 230 else \
                    "P300 (Late Positive)" if lat == 350 else \
                    f"{lat} ms"
        im = plot_scalp_topomap(axes[i], vals, x_coords, y_coords, vmin=v_min, vmax=v_max,
                                title=f"{lat} ms\n{title_tag}")

    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.22, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Voltage Amplitude (μV)", fontsize=10, fontweight="bold")
    plt.savefig(os.path.join(FIG_DIR, "fig3_scalp_topographies.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved: fig3_scalp_topographies.png")

    # =========================================================================
    # Figure 4: Cross-Subject ERP & SNR Comparison
    # =========================================================================
    print("\n[Phase 5] Generating Figure 4: Cross-Subject ERP Comparison & SNR Profiles...")
    fig = plt.figure(figsize=(15, 9), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 1.0])

    ax_oz = fig.add_subplot(gs[0, 0])
    ax_po = fig.add_subplot(gs[1, 0], sharex=ax_oz)
    ax_snr = fig.add_subplot(gs[0, 1])
    ax_peaks = fig.add_subplot(gs[1, 1])

    # Plot Oz and PO7/8 ERPs for all 10 subjects
    colors_10 = plt.cm.tab10(np.linspace(0, 1, 10))
    for sub in ALL_SUBJECTS:
        color = "crimson" if sub == 9 else "darkgreen" if sub == 8 else "royalblue" if sub == 3 else colors_10[sub - 1]
        lw = 2.5 if sub in [8, 3, 9] else 1.0
        alpha = 0.95 if sub in [8, 3, 9] else 0.4
        label = f"Sub-{sub:02d} (Low SNR)" if sub == 9 else \
                f"Sub-{sub:02d} (High SNR)" if sub == 8 else \
                f"Sub-{sub:02d} (High SNR)" if sub == 3 else f"Sub-{sub:02d}"

        oz_trace = all_erps[sub - 1, ch_dict["Oz"]]
        po_trace = all_erps[sub - 1, roi_indices["Lateral Parieto-Occipital (PO7, PO8)"]].mean(axis=0)

        ax_oz.plot(times_ms, oz_trace, color=color, linewidth=lw, alpha=alpha, label=label if sub in [8, 3, 9] else None)
        ax_po.plot(times_ms, po_trace, color=color, linewidth=lw, alpha=alpha)

    ax_oz.axvline(0, color="black", linestyle="--")
    ax_oz.axhline(0, color="black", linestyle=":")
    ax_oz.set_ylabel("Oz Amplitude (μV)", fontweight="bold")
    ax_oz.set_title("Electrode Oz: Occipital Evoked Response Across Subjects", fontweight="bold")
    ax_oz.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax_oz.grid(True, linestyle=":", alpha=0.5)

    ax_po.axvline(0, color="black", linestyle="--")
    ax_po.axhline(0, color="black", linestyle=":")
    ax_po.set_xlabel("Time (ms)", fontweight="bold")
    ax_po.set_ylabel("PO7/PO8 Mean (μV)", fontweight="bold")
    ax_po.set_title("Electrodes PO7/PO8: Lateral Occipito-Temporal Response (N170)", fontweight="bold")
    ax_po.grid(True, linestyle=":", alpha=0.5)

    # Bar chart: Evoked SNR Ratio across all 10 subjects
    sub_labels = [f"Sub-{s:02d}" for s in ALL_SUBJECTS]
    snrs = df_snr["evoked_snr_x1000"].values
    bar_colors = ["crimson" if s == 9 else "darkgreen" if s == 8 else "royalblue" if s == 3 else "steelblue" for s in ALL_SUBJECTS]
    bars = ax_snr.bar(sub_labels, snrs, color=bar_colors, alpha=0.85, edgecolor="black")
    ax_snr.set_ylabel("Evoked SNR (Ratio x 10³)", fontweight="bold")
    ax_snr.set_title("Evoked Signal-to-Noise Ratio (Evoked Power / Trial Variance)", fontweight="bold")
    ax_snr.grid(axis="y", linestyle=":", alpha=0.6)
    ax_snr.tick_params(axis="x", rotation=40)
    for bar, val in zip(bars, snrs):
        ax_snr.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")

    # Bar chart: P100 vs N170 amplitude across subjects
    w = 0.35
    x = np.arange(10)
    ax_peaks.bar(x - w / 2, df_peaks["P100_amp_uv"], w, label="P100 Max (Parieto-Occipital)", color="darkorange", alpha=0.85)
    ax_peaks.bar(x + w / 2, np.abs(df_peaks["N170_amp_uv"]), w, label="|N170| Amplitude (PO7/8)", color="purple", alpha=0.85)
    ax_peaks.set_xticks(x)
    ax_peaks.set_xticklabels(sub_labels, rotation=40)
    ax_peaks.set_ylabel("Peak Amplitude (μV)", fontweight="bold")
    ax_peaks.set_title("P100 (PO) and |N170| (PO7/8) Peak Magnitude per Subject", fontweight="bold")
    ax_peaks.grid(axis="y", linestyle=":", alpha=0.6)
    ax_peaks.legend(loc="upper right", fontsize=8)

    plt.suptitle("Cross-Subject Signal Fidelity: Neurophysiological Grounding of RDM Reliability", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_subject_erp_snr_comparison.png"), dpi=300)
    plt.close()
    print("  -> Saved: fig4_subject_erp_snr_comparison.png")

    # =========================================================================
    # Phase 6: Time-Frequency Spectrograms (ERSP & ITPC)
    # =========================================================================
    print("\n[Phase 6] Computing Time-Frequency Representation (ERSP & ITPC via Morlet Wavelets)...")
    freqs = np.linspace(4, 45, 42)  # 4 Hz to 45 Hz

    # Load high-SNR Subject 08 and low-SNR Subject 09 single-trial data for Occipital ROI
    sub8_data, _ = load_subject_data(8)
    sub9_data, _ = load_subject_data(9)

    occ_indices = roi_indices["Occipital (O1, Oz, O2)"]
    # Sample 400 single trials for fast, stable single-trial ITPC / ERSP calculation
    # (200 concepts x 2 reps = 400 trials)
    sub8_occ = sub8_data[:, :2, occ_indices, :].reshape(-1, 100)  # (400*3 = 1200 trials)
    sub9_occ = sub9_data[:, :2, occ_indices, :].reshape(-1, 100)

    # Compute continuous Morlet CWT
    cwt_sub8 = morlet_cwt(sub8_occ, sr=SR, freqs=freqs, n_cycles=5)  # (N_trials, n_freqs, n_times)
    cwt_sub9 = morlet_cwt(sub9_occ, sr=SR, freqs=freqs, n_cycles=5)

    # 1. Total Power
    p_total_8 = np.mean(np.abs(cwt_sub8)**2, axis=0)  # (n_freqs, n_times)
    p_total_9 = np.mean(np.abs(cwt_sub9)**2, axis=0)

    # 2. Baseline Normalization: Event-Related Spectral Perturbation (ERSP in dB)
    base_p_8 = p_total_8[:, BASELINE_SLICE].mean(axis=-1, keepdims=True)
    base_p_9 = p_total_9[:, BASELINE_SLICE].mean(axis=-1, keepdims=True)

    ersp_8 = 10.0 * np.log10(p_total_8 / (base_p_8 + 1e-12))
    ersp_9 = 10.0 * np.log10(p_total_9 / (base_p_9 + 1e-12))

    # 3. Inter-Trial Phase Coherence (ITPC / Phase-Locking Value)
    # ITPC = |(1/N) * sum(e^{i phi})|
    phase_8 = np.exp(1j * np.angle(cwt_sub8))
    itpc_8 = np.abs(np.mean(phase_8, axis=0))

    phase_9 = np.exp(1j * np.angle(cwt_sub9))
    itpc_9 = np.abs(np.mean(phase_9, axis=0))

    # Figure 5: ERSP and ITPC Spectrograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300, sharex=True, sharey=True)

    # Common colorbar scale for ERSP
    ersp_max = np.percentile(np.abs(ersp_8), 98)
    ersp_im = axes[0, 0].pcolormesh(times_ms, freqs, ersp_8, cmap="RdBu_r", vmin=-ersp_max, vmax=ersp_max, shading="gouraud")
    axes[0, 0].set_title("Sub-08 (High SNR): Occipital ERSP (dB re: baseline)", fontweight="bold")
    axes[0, 0].set_ylabel("Frequency (Hz)", fontweight="bold")
    axes[0, 0].axvline(0, color="black", linestyle="--")
    cb0 = fig.colorbar(ersp_im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb0.set_label("Power Change (dB)", fontsize=9)

    axes[0, 1].pcolormesh(times_ms, freqs, ersp_9, cmap="RdBu_r", vmin=-ersp_max, vmax=ersp_max, shading="gouraud")
    axes[0, 1].set_title("Sub-09 (Low SNR): Occipital ERSP (dB re: baseline)", fontweight="bold")
    axes[0, 1].axvline(0, color="black", linestyle="--")
    cb1 = fig.colorbar(ersp_im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb1.set_label("Power Change (dB)", fontsize=9)

    # ITPC plots
    itpc_im = axes[1, 0].pcolormesh(times_ms, freqs, itpc_8, cmap="viridis", vmin=0, vmax=0.6, shading="gouraud")
    axes[1, 0].set_title("Sub-08: Inter-Trial Phase Coherence (ITPC)", fontweight="bold")
    axes[1, 0].set_xlabel("Time Relative to Stimulus Onset (ms)", fontweight="bold")
    axes[1, 0].set_ylabel("Frequency (Hz)", fontweight="bold")
    axes[1, 0].axvline(0, color="white", linestyle="--")
    cb2 = fig.colorbar(itpc_im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb2.set_label("Phase Locking (0–1)", fontsize=9)

    axes[1, 1].pcolormesh(times_ms, freqs, itpc_9, cmap="viridis", vmin=0, vmax=0.6, shading="gouraud")
    axes[1, 1].set_title("Sub-09: Inter-Trial Phase Coherence (ITPC)", fontweight="bold")
    axes[1, 1].set_xlabel("Time Relative to Stimulus Onset (ms)", fontweight="bold")
    axes[1, 1].axvline(0, color="white", linestyle="--")
    cb3 = fig.colorbar(itpc_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb3.set_label("Phase Locking (0–1)", fontsize=9)

    # Add frequency band indicator lines
    for ax in axes.flatten():
        ax.axhline(8, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(12, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(30, color="gray", linestyle=":", alpha=0.6)

    plt.suptitle("Time-Frequency Dynamics: Evoked Spectral Perturbation & Phase Resetting", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_time_frequency_ersp_itpc.png"), dpi=300)
    plt.close()
    print("  -> Saved: fig5_time_frequency_ersp_itpc.png")

    # =========================================================================
    # Phase 7: Power Spectral Density (PSD) & Alpha Desynchronization (ERD)
    # =========================================================================
    print("\n[Phase 7] Computing Power Spectral Density (PSD) & Alpha Desynchronization...")
    # Calculate PSD using Welch's method across all subjects at Occipital channels
    bands = {
        "Delta (1–4 Hz)": (1, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–12 Hz)": (8, 12),
        "Beta (13–30 Hz)": (13, 30),
        "Low Gamma (30–45 Hz)": (30, 45),
    }

    band_records = []
    # Collect PSD curves
    psd_pre_all = []
    psd_post_all = []
    freq_axis = None

    for sub in ALL_SUBJECTS:
        # Load raw single trials at occipital channels (200 concepts x 80 reps = 16,000 trials x 3 channels)
        # Sliced: baseline 20 samples, post-stimulus 80 samples
        data, _ = load_subject_data(sub)
        occ_data = data[:, :, roi_indices["Occipital (O1, Oz, O2)"], :]  # (200, 80, 3, 100)
        occ_flat = occ_data.reshape(-1, 100)

        # Baseline: pad 20 samples to 100 for consistent frequency resolution
        pre_sig = occ_flat[:, BASELINE_SLICE]
        post_sig = occ_flat[:, POSTSTIM_SLICE]

        f_pre, p_pre = signal.welch(pre_sig, fs=SR, nperseg=20, nfft=100, axis=-1)
        f_post, p_post = signal.welch(post_sig, fs=SR, nperseg=80, nfft=100, axis=-1)
        freq_axis = f_post

        p_pre_mean = np.mean(p_pre, axis=0)
        p_post_mean = np.mean(p_post, axis=0)
        psd_pre_all.append(p_pre_mean)
        psd_post_all.append(p_post_mean)

        for b_name, (f_lo, f_hi) in bands.items():
            b_mask_pre = (f_pre >= f_lo) & (f_pre <= f_hi)
            b_mask_post = (f_post >= f_lo) & (f_post <= f_hi)
            bp_pre = float(np.trapz(p_pre_mean[b_mask_pre], f_pre[b_mask_pre]))
            bp_post = float(np.trapz(p_post_mean[b_mask_post], f_post[b_mask_post]))
            erd_pct = ((bp_post - bp_pre) / (bp_pre + 1e-12)) * 100.0

            band_records.append({
                "subject": sub,
                "band": b_name,
                "power_pre": round(bp_pre, 4),
                "power_post": round(bp_post, 4),
                "erd_pct": round(erd_pct, 2),
            })

    df_bands = pd.DataFrame(band_records)
    df_bands.to_csv(os.path.join(TAB_DIR, "frequency_band_power_and_erd.csv"), index=False)

    psd_pre_mean = np.mean(psd_pre_all, axis=0)
    psd_post_mean = np.mean(psd_post_all, axis=0)

    # Figure 6: PSD Curves & Alpha ERD
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # PSD spectra
    axes[0].plot(freq_axis, 10 * np.log10(psd_pre_mean + 1e-12), color="royalblue", lw=2.5, label="Pre-stimulus Baseline (-200 to 0 ms)")
    axes[0].plot(freq_axis, 10 * np.log10(psd_post_mean + 1e-12), color="crimson", lw=2.5, label="Post-stimulus Evoked (0 to 790 ms)")
    axes[0].axvspan(8, 12, color="gold", alpha=0.25, label="Alpha Band (8–12 Hz)")
    axes[0].set_xlim(1, 45)
    axes[0].set_xlabel("Frequency (Hz)", fontweight="bold")
    axes[0].set_ylabel("Power Spectral Density (dB/Hz)", fontweight="bold")
    axes[0].set_title("Occipital Power Spectral Density (N=10 Subjects)", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Band-wise ERD/ERS Bar Chart (Grand Average across subjects)
    erd_summary = df_bands.groupby("band")["erd_pct"].agg(["mean", "sem"]).reindex(list(bands.keys()))
    x_b = np.arange(len(erd_summary))
    erd_colors = ["forestgreen" if val > 0 else "firebrick" for val in erd_summary["mean"]]
    b_bars = axes[1].bar(x_b, erd_summary["mean"], yerr=erd_summary["sem"], color=erd_colors, alpha=0.85, edgecolor="black", capsize=4)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=1)
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels([b.split(" ")[0] for b in erd_summary.index], fontweight="bold")
    axes[1].set_ylabel("Event-Related Desynchronization (ERD %)", fontweight="bold")
    axes[1].set_title("Frequency Band Power Modulation (% Change re: Baseline)", fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    for bar, val in zip(b_bars, erd_summary["mean"]):
        y_pos = val + (2 if val > 0 else -5)
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Spectral Signatures: Alpha Desynchronization (ERD) & Evoked Rhythms", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_psd_and_alpha_desync.png"), dpi=300)
    plt.close()
    print("  -> Saved: fig6_psd_and_alpha_desync.png")

    # =========================================================================
    # Figure 7: Integrated Executive Signal Analysis Dashboard
    # =========================================================================
    print("\n[Phase 8] Generating Integrated Executive Signal Analysis Dashboard...")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1.0, 1.0])

    # Panel A: Butterfly & GFP
    ax_a = fig.add_subplot(gs[0, :2])
    for ch_i in range(63):
        color = "crimson" if ch_i in roi_indices["Occipital (O1, Oz, O2)"] else \
                "royalblue" if ch_i in roi_indices["Lateral Parieto-Occipital (PO7, PO8)"] else "lightgray"
        ax_a.plot(times_ms, grand_erp[ch_i], color=color, alpha=0.6 if color != "lightgray" else 0.3, lw=1.2)
    ax_a.plot(times_ms, grand_gfp, color="black", lw=2.5, label="Global Field Power (GFP)")
    ax_a.axvline(0, color="black", linestyle="--")
    ax_a.axvline(110, color="gold", linestyle=":", lw=1.5, label="P100 (110 ms)")
    ax_a.axvline(170, color="darkorange", linestyle=":", lw=1.5, label="N170 (170 ms)")
    ax_a.set_ylabel("Amplitude (μV)", fontweight="bold")
    ax_a.set_title("(A) Grand-Average ERP Butterfly Plot & Global Field Power (N=10 Subjects)", fontweight="bold")
    ax_a.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    # Panel B: P100 Scalp Map
    ax_b1 = fig.add_subplot(gs[0, 2])
    plot_scalp_topomap(ax_b1, grand_erp[:, sample_indices[1]], x_coords, y_coords,
                       vmin=v_min, vmax=v_max, title="(B1) P100 Peak (110 ms)\nOccipital Positive Pole")

    # Panel C: N170 Scalp Map
    ax_b2 = fig.add_subplot(gs[1, 2])
    plot_scalp_topomap(ax_b2, grand_erp[:, sample_indices[2]], x_coords, y_coords,
                       vmin=v_min, vmax=v_max, title="(B2) N170 Peak (170 ms)\nBilateral Lateral Negativity")

    # Panel D: ROI Waveforms
    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.plot(times_ms, grand_erp[ch_dict["Oz"]], color="crimson", lw=2.5, label="Oz (Occipital P100/N170)")
    ax_d.plot(times_ms, grand_erp[roi_indices["Lateral Parieto-Occipital (PO7, PO8)"]].mean(axis=0), color="royalblue", lw=2.5, label="PO7/PO8 (Lateral N170)")
    ax_d.plot(times_ms, grand_erp[ch_dict["Cz"]], color="darkgreen", lw=2, label="Cz (Vertex Positive)")
    ax_d.axvline(0, color="black", linestyle="--")
    ax_d.axhline(0, color="black", linestyle=":")
    ax_d.set_ylabel("Amplitude (μV)", fontweight="bold")
    ax_d.set_title("(C) Regional Evoked Waveforms (Canonical Electrode Montages)", fontweight="bold")
    ax_d.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    # Panel E: ERSP Spectrogram
    ax_e = fig.add_subplot(gs[2, 0])
    im_e = ax_e.pcolormesh(times_ms, freqs, ersp_8, cmap="RdBu_r", vmin=-ersp_max, vmax=ersp_max, shading="gouraud")
    ax_e.set_title("(D) Occipital ERSP Spectrogram (Sub-08)", fontweight="bold")
    ax_e.set_xlabel("Time (ms)", fontweight="bold")
    ax_e.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax_e.axvline(0, color="black", linestyle="--")
    fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)

    # Panel F: ITPC Spectrogram
    ax_f = fig.add_subplot(gs[2, 1])
    im_f = ax_f.pcolormesh(times_ms, freqs, itpc_8, cmap="viridis", vmin=0, vmax=0.6, shading="gouraud")
    ax_f.set_title("(E) Inter-Trial Phase Coherence (ITPC)", fontweight="bold")
    ax_f.set_xlabel("Time (ms)", fontweight="bold")
    ax_f.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax_f.axvline(0, color="white", linestyle="--")
    fig.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)

    # Panel G: Evoked SNR per Subject
    ax_g = fig.add_subplot(gs[2, 2])
    ax_g.bar(sub_labels, snrs, color=bar_colors, alpha=0.85, edgecolor="black")
    ax_g.set_title("(F) Evoked SNR Ratio x 10³", fontweight="bold")
    ax_g.set_ylabel("Evoked SNR", fontweight="bold")
    ax_g.tick_params(axis="x", rotation=45)
    ax_g.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Executive EEG Signal Analysis Dashboard: THINGS-EEG2 Neurophysiological Grounding", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    dashboard_path = os.path.join(FIG_DIR, "fig7_executive_signal_dashboard.png")
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    print(f"  -> Saved Executive Dashboard: {dashboard_path}")

    # Summary JSON
    elapsed = time.time() - t_start
    summary_report = {
        "dataset": "THINGS-EEG2 (Test partition: 200 concepts x 80 reps)",
        "n_subjects": 10,
        "n_channels": 63,
        "sampling_rate_hz": 100,
        "grand_average_p100": {
            "latency_ms": float(df_peaks["P100_lat_ms"].mean()),
            "amplitude_uv": float(df_peaks["P100_amp_uv"].mean()),
        },
        "grand_average_n170": {
            "latency_ms": float(df_peaks["N170_lat_ms"].mean()),
            "amplitude_uv": float(df_peaks["N170_amp_uv"].mean()),
        },
        "grand_average_p300": {
            "latency_ms": float(df_peaks["P300_lat_ms"].mean()),
            "amplitude_uv": float(df_peaks["P300_amp_uv"].mean()),
        },
        "evoked_snr_ranking": df_snr.sort_values(by="evoked_snr_ratio", ascending=False)[["subject", "evoked_snr_x1000"]].to_dict(orient="records"),
        "concept_snr_ranking": df_snr.sort_values(by="concept_snr_ratio", ascending=False)[["subject", "concept_snr_x1000"]].to_dict(orient="records"),
        "alpha_erd_percent": float(erd_summary.loc["Alpha (8–12 Hz)", "mean"]),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(os.path.join(TAB_DIR, "signal_analysis_summary.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Signal Analysis Pipeline Successfully Completed in {elapsed:.1f} s")
    print(f"All figures saved to: {FIG_DIR}")
    print(f"All tables saved to:  {TAB_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_eeg_signal_analysis()

