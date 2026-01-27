#!/usr/bin/env python3
# plot_figure1_pipeline.py (MATCHED + FALLBACK SAFE)
# ------------------------------------------------------------
# MỤC TIÊU: Xuất các panel PDF cho Figure 1.
# YÊU CẦU: Các hàm xử lý phải giống ecg_feature_extractor.py:
#   - detect_rpeaks: nk.ecg_clean -> nk.ecg_peaks (không correct_artifacts, không fixpeaks)
#   - compute_rri: diff(rpeaks/fs)
#   - interpolate_rri: quadratic @4Hz + robust NaN handling
#   - segment_rri: non-overlap windows (window_sec * 4Hz)
#   - extract_hrv_features: nk.hrv({"RRI","RRI_Time"}, sampling_rate=fs)
#
# FIX thực tế cho Figure:
#   - Nếu record không đủ 300s để tạo segment: fallback sang window ngắn nhất có thể
#     (vẫn giữ fs_interp=4Hz, kind=quadratic, chỉ thay window hiệu dụng để script không crash).
# ------------------------------------------------------------

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib.patches import FancyArrowPatch


# =============================
# CONFIG
# =============================
path_ecg = r"..\..\..\data\raw\0001"

# Nếu dataset của anh là Lead II ở index=1 thì giữ 1.
# Nếu record chỉ có 1 channel, code sẽ tự fallback về 0.
lead_idx = 1

SKIP_SAMPLES = 0

# Tham số phải giống extractor
hrv_window_sec = 300
fs_interp = 4.0
interp_kind = "quadratic"

# Figure spec: 10s snippet (anh có thể đổi về 5 nếu muốn)
demo_duration_sec = 10
demo_start_sec = 0

OUT_DIR = "pdf"
BASE_FONT = 12

# Fallback guard: nếu record quá ngắn (HRV không có ý nghĩa), dừng.
MIN_HRV_SEC = 60  # có thể hạ 30 nếu record quá ngắn, nhưng 60s an toàn hơn


# =============================
# FUNCTIONS (COPIED FROM extractor)
# =============================

def detect_rpeaks(raw_ecg, fs):
    ecg_clean = nk.ecg_clean(raw_ecg, sampling_rate=fs)
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    return info["ECG_R_Peaks"]


def compute_rri(rpeaks, fs):
    times = np.array(rpeaks) / fs
    rri = np.diff(times)
    times_rri = times[1:]
    return times_rri, rri


def interpolate_rri(times_rri, rri, fs_interp=4.0, kind="quadratic"):
    times_rri = np.asarray(times_rri, dtype=float)
    rri = np.asarray(rri, dtype=float)

    # bỏ NaN/inf trước khi nội suy
    m = np.isfinite(times_rri) & np.isfinite(rri)
    times_rri = times_rri[m]
    rri = rri[m]

    if times_rri.size < 2:
        return np.array([]), np.array([])

    # Lưới thời gian đều
    t_interp = np.arange(times_rri[0], times_rri[-1] + 1e-12, 1.0 / fs_interp)

    # Nội suy "an toàn": không lỗi biên, điền bằng ngoại suy tuyến tính
    f = interp1d(
        times_rri, rri,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    y = f(t_interp)

    # Nếu còn NaN hiếm, điền tuyến tính ngắn
    if np.any(~np.isfinite(y)):
        good = np.isfinite(y)
        if good.any():
            y[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), y[good])
        else:
            return np.array([]), np.array([])

    return t_interp, y


def segment_rri(t_interp, rri_interp, fs_interp=4.0, window_sec=300):
    window_size = int(window_sec * fs_interp)
    segments = []
    for start in range(0, len(rri_interp) - window_size + 1, window_size):
        segments.append(rri_interp[start:start + window_size])
    return segments


def extract_hrv_features(rri_segments, fs):
    rows = []
    for rri in rri_segments:
        time = np.linspace(0, len(rri)/4.0, len(rri))
        hrv_all = nk.hrv({"RRI": rri*1000, "RRI_Time": time}, sampling_rate=fs, show=False)
        rows.append(hrv_all.iloc[0].to_dict())

    df = pd.DataFrame(rows)
    cols = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_HTI", "HRV_TINN",
        "HRV_VLF", "HRV_LF", "HRV_LFn", "HRV_HF", "HRV_HFn", "HRV_LFHF", "HRV_TP",
        "HRV_ApEn", "HRV_SampEn", "HRV_DFA_alpha1", "HRV_DFA_alpha2", "HRV_CD",
        "HRV_SD1", "HRV_SD2"
    ]
    df20 = df[cols]
    return df20


# =============================
# PLOT HELPERS
# =============================

def set_plot_style(base_font: int = 12):
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 2,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.30,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_pdf(fig, filepath: str, extra_artists=None):
    kwargs = {}
    if extra_artists:
        kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)
    print(f"✓ Saved {filepath}")


def debug_checks(demo_signal, demo_peaks_indices, demo_peak_times, demo_peak_values, fs, stage_name="Demo Slice"):
    print(f"\n🔍 --- DEBUG REPORT: {stage_name} ---")

    n_samples = len(demo_signal)
    n_peaks = len(demo_peaks_indices)
    print(f"   [Info] Signal duration: {n_samples/fs:.2f}s ({n_samples} samples)")
    print(f"   [Info] Peaks detected: {n_peaks}")

    if n_peaks == 0:
        print("   ⚠️ [WARN] Không tìm thấy đỉnh R nào trong đoạn này!")
        return

    min_idx = int(np.min(demo_peaks_indices))
    max_idx = int(np.max(demo_peaks_indices))

    if min_idx < 0 or max_idx >= n_samples:
        print(f"   ❌ [FAIL] Peak index out of bounds [0, {n_samples-1}]")
        print(f"      Min peak: {min_idx}, Max peak: {max_idx}")
    else:
        print("   ✅ [PASS] Bounds check: Peaks within window.")

    signal_values_at_peaks = demo_signal[demo_peaks_indices]
    diff = np.abs(signal_values_at_peaks - demo_peak_values)
    max_diff = float(np.max(diff))

    if max_diff > 1e-6:
        print("   ❌ [FAIL] Peak alignment mismatch (scatter not on curve).")
        print(f"      Max abs diff: {max_diff:.6f}")
    else:
        print("   ✅ [PASS] Alignment check: scatter matches curve.")

    calculated_times = demo_peaks_indices / fs
    time_diff = np.abs(calculated_times - demo_peak_times)
    if float(np.max(time_diff)) > 1e-6:
        print("   ❌ [FAIL] Time axis mismatch.")
    else:
        print("   ✅ [PASS] Time axis OK.")
    print("--------------------------------------------------\n")


# =============================
# MAIN
# =============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_plot_style(BASE_FONT)

    print(f"Loading record: {path_ecg} (Lead index: {lead_idx})")
    try:
        record = wfdb.rdrecord(path_ecg)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find file at {path_ecg}")
        return

    fs = float(record.fs)

    if record.p_signal is None or record.p_signal.ndim != 2:
        raise RuntimeError("record.p_signal is missing or not 2D. Check WFDB record format.")

    n_ch = record.p_signal.shape[1]
    if lead_idx < 0 or lead_idx >= n_ch:
        print(f"[Warn] lead_idx={lead_idx} out of range (record has {n_ch} channel). Fallback lead_idx=0.")
        use_lead = 0
    else:
        use_lead = lead_idx

    raw_ecg_full = record.p_signal[:, use_lead]

    if SKIP_SAMPLES > 0:
        if SKIP_SAMPLES >= len(raw_ecg_full):
            raise RuntimeError(f"SKIP_SAMPLES={SKIP_SAMPLES} >= signal length {len(raw_ecg_full)}.")
        raw_ecg_full = raw_ecg_full[SKIP_SAMPLES:]

    # Main window = 300s nếu đủ, else full record (giống logic an toàn của figure)
    n_main = int(hrv_window_sec * fs)
    if len(raw_ecg_full) >= n_main:
        raw_main = raw_ecg_full[:n_main]
    else:
        raw_main = raw_ecg_full
        print(f"[Warn] Record shorter than {hrv_window_sec}s. Using full length ({len(raw_main)/fs:.1f}s).")

    # ---- Pipeline giống extractor ----
    # Peaks
    rpeaks_main = np.array(detect_rpeaks(raw_main, fs), dtype=int)

    # Clean for plotting (cùng thao tác với detect_rpeaks)
    ecg_clean_main = nk.ecg_clean(raw_main, sampling_rate=fs)

    if len(rpeaks_main) < 10:
        raise RuntimeError(f"Too few R-peaks detected in main window: {len(rpeaks_main)}.")

    # RRI raw
    times_rri_main, rri_main = compute_rri(rpeaks_main, fs)

    # Interp 4Hz
    t_interp_main, rri_interp_main = interpolate_rri(times_rri_main, rri_main, fs_interp=fs_interp, kind=interp_kind)
    if t_interp_main.size == 0 or rri_interp_main.size == 0:
        raise RuntimeError("Interpolation returned empty result (not enough valid RR intervals).")

    # Segment 300s (extractor style)
    rri_segs = segment_rri(t_interp_main, rri_interp_main, fs_interp=fs_interp, window_sec=hrv_window_sec)

    # ✅ FIX: fallback nếu không đủ 300s để tạo segment
    if len(rri_segs) == 0:
        available_sec = float(len(rri_interp_main) / fs_interp)
        eff_sec = int(np.floor(available_sec))
        print(f"[Warn] Not enough interpolated RRI length for 300s@4Hz (need 1200 samples). "
              f"Available ≈ {available_sec:.1f}s.")

        if eff_sec < MIN_HRV_SEC:
            raise RuntimeError(
                f"Record too short for HRV fallback: only ~{available_sec:.1f}s interpolated RRI. "
                f"Need at least {MIN_HRV_SEC}s."
            )

        # tạo 1 segment fallback (vẫn dùng đúng RRI nội suy 4Hz và nk.hrv như extractor)
        eff_len = int(eff_sec * fs_interp)
        rri_segs = [rri_interp_main[:eff_len]]
        print(f"[Warn] Using HRV fallback window: {eff_sec}s (segment length={eff_len} samples).")

    # HRV features (extractor style)
    hrv_df20 = extract_hrv_features(rri_segs, fs=fs)

    # ---- Demo snippet (10s) for plots ----
    start_idx = int(demo_start_sec * fs)
    n_demo = int(demo_duration_sec * fs)
    end_idx = start_idx + n_demo

    if end_idx > len(ecg_clean_main):
        print("[Warn] Demo segment out of bounds. Adjusting to end of record.")
        end_idx = len(ecg_clean_main)
        start_idx = max(0, end_idx - n_demo)

    raw_demo = raw_main[start_idx:end_idx]
    clean_demo = ecg_clean_main[start_idx:end_idx]
    t_demo = np.arange(len(clean_demo)) / fs

    # Peaks in demo (pixel-aligned)
    mask_peaks = (rpeaks_main >= start_idx) & (rpeaks_main < end_idx)
    rpeaks_demo_indices = (rpeaks_main[mask_peaks] - start_idx).astype(int)
    peak_times_demo = rpeaks_demo_indices / fs
    peak_values_demo = clean_demo[rpeaks_demo_indices] if len(rpeaks_demo_indices) > 0 else np.array([])

    debug_checks(
        demo_signal=clean_demo,
        demo_peaks_indices=rpeaks_demo_indices,
        demo_peak_times=peak_times_demo,
        demo_peak_values=peak_values_demo if len(peak_values_demo) > 0 else np.array([]),
        fs=fs,
        stage_name="Panel C Visualization"
    )

    # RRI cut for demo (absolute sec in main window)
    demo_start_t = start_idx / fs
    demo_end_t = end_idx / fs

    mask_rri = (times_rri_main >= demo_start_t) & (times_rri_main < demo_end_t)
    t_rri_demo = times_rri_main[mask_rri] - demo_start_t
    val_rri_demo = rri_main[mask_rri]

    mask_interp = (t_interp_main >= demo_start_t) & (t_interp_main < demo_end_t)
    t_int_demo = t_interp_main[mask_interp] - demo_start_t
    val_int_demo = rri_interp_main[mask_interp]

    x_max = len(clean_demo) / fs

    # =============================
    # EXPORT PANELS
    # =============================

    # Panel A
    figA, axA = plt.subplots(figsize=(6, 2.1))
    axA.plot(t_demo, raw_demo, color="#555555", linewidth=0.9)
    axA.set_xlim(0, x_max)
    axA.set_ylabel("Voltage (mV)")
    axA.set_xlabel("Time (s)")
    axA.set_title(r"$x_{\mathrm{raw}}(t)$")
    axA.grid(True)
    save_pdf(figA, os.path.join(OUT_DIR, "panel_A_raw.pdf"))

    # Panel B
    figB, axB = plt.subplots(figsize=(6, 2.1))
    axB.plot(t_demo, clean_demo, color="#1f77b4", linewidth=0.9)
    axB.set_xlim(0, x_max)
    axB.set_ylabel("Voltage (mV)")
    axB.set_xlabel("Time (s)")
    axB.set_title(r"$x_{\mathrm{clean}}(t)$")
    axB.grid(True)
    save_pdf(figB, os.path.join(OUT_DIR, "panel_B_clean.pdf"))

    # Panel C
    figC, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 4.4),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35}
    )

    ax1.plot(t_demo, clean_demo, color="#1f77b4", linewidth=0.9)
    if len(peak_times_demo) > 0:
        ax1.scatter(peak_times_demo, peak_values_demo, color="red", s=25, zorder=5, label="R-peaks")
    ax1.set_xlim(0, x_max)
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("R-peak Detection")
    ax1.grid(True)

    ax2.step(t_rri_demo, val_rri_demo, where="pre", color="gray", alpha=0.6, label="Raw RRI", linewidth=1.2)
    ax2.plot(t_int_demo, val_int_demo, color="#d62728", linewidth=1.5, label="4 Hz Interpolated")
    ax2.set_xlim(0, x_max)
    ax2.set_ylabel("RRI (s)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    figC.subplots_adjust(right=0.78)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = figC.legend(
        h1 + h2, l1 + l2,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        bbox_transform=figC.transFigure,
        frameon=False,
        fontsize=10
    )

    p1 = ax1.get_position()
    p2 = ax2.get_position()
    arrow_x = (p1.x0 + p1.x1) / 2
    arrow = FancyArrowPatch(
        (arrow_x, p1.y0), (arrow_x, p2.y1),
        transform=figC.transFigure,
        arrowstyle="->", mutation_scale=15, lw=1.5, color="black"
    )
    figC.add_artist(arrow)

    save_pdf(figC, os.path.join(OUT_DIR, "panel_C_features.pdf"), extra_artists=[leg])

    # Panel HRV table (from extractor-style HRV df20, first segment)
    cols_wanted = ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_MeanNN", "HRV_LFHF"]
    row0 = hrv_df20.iloc[0]

    table_data = []
    for col in cols_wanted:
        if col in row0.index and pd.notna(row0[col]):
            val = float(row0[col])
            key_nice = col.replace("HRV_", "")
            if key_nice in ["SDNN", "RMSSD", "MeanNN"]:
                val_str = f"{val:.1f}"
            elif key_nice == "pNN50":
                val_str = f"{val:.1f}"
            else:
                val_str = f"{val:.2f}"
            table_data.append([key_nice, val_str])
        else:
            table_data.append([col.replace("HRV_", ""), "-"])

    df_tbl = pd.DataFrame(table_data, columns=["Feature", "Value"])

    figT, axT = plt.subplots(figsize=(3.5, 2.2))
    axT.axis("off")
    table = axT.table(
        cellText=df_tbl.values,
        colLabels=df_tbl.columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.5, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    save_pdf(figT, os.path.join(OUT_DIR, "panel_C_hrv_table.pdf"))

    print("\n✅ DONE! All PDF panels saved in 'pdf/' folder.")


if __name__ == "__main__":
    main()
