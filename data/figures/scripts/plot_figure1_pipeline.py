# plot_figure1_pipeline.py (FINAL FIXED VERSION)
# ------------------------------------------------------------
# MỤC TIÊU: Tạo ra các mảnh (Panels) cho Figure 1 chuẩn mực Journal.
#
# QUY TRÌNH XỬ LÝ:
#   1. Load dữ liệu ECG thô.
#   2. Chạy pipeline chuẩn trên cửa sổ lớn (ví dụ: 300s) để tính toán chính xác:
#      - Lọc nhiễu (NeuroKit2 ecg_clean).
#      - Tìm đỉnh R (R-peaks) + sửa artefact (nếu hỗ trợ).
#      - Tính chuỗi RRI và nội suy 4Hz.
#      - Tính bảng chỉ số HRV.
#   3. Cắt một đoạn ngắn (ví dụ: 20s) từ kết quả trên để VẼ MINH HỌA (Figure 1).
#      - Chấm đỏ (R-peaks) khớp pixel với sóng ECG (lấy đúng clean_demo[idx]).
#      - RRI thô + nội suy khớp pha thời gian (cắt theo time mask).
#   4. FIX layout: legend/figure label nằm NGOÀI vùng biểu đồ (axes) ở Panel C.
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
# CONFIG (Cấu hình đường dẫn và thông số)
# =============================
path_ecg = r"..\..\..\data\raw\0001"
lead_idx = 1  # Lead II thường là 1, tùy dataset

SKIP_SAMPLES = 0

hrv_window_sec = 300
fs_interp = 4.0

demo_duration_sec = 5
demo_start_sec = 0

OUT_DIR = "pdf"
BASE_FONT = 12

PEAK_METHOD = "neurokit"
CORRECT_ARTIFACTS = True
FIXPEAKS_METHOD = "Kubios"


# =============================
# HÀM HỖ TRỢ
# =============================

def set_plot_style(base_font: int = 12):
    """Thiết lập style cho Matplotlib để hình ảnh sắc nét và chuẩn font."""
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
    """Lưu figure ra file PDF với lề gọn gàng (hỗ trợ legend nằm ngoài axes)."""
    kwargs = {}
    if extra_artists:
        kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)
    print(f"✓ Saved {filepath}")


def detect_rpeaks_robust(ecg_clean: np.ndarray, fs: float) -> np.ndarray:
    """Phát hiện đỉnh R có sửa lỗi (Artifact Correction), tương thích nhiều phiên bản NeuroKit2."""
    if CORRECT_ARTIFACTS:
        try:
            _, info = nk.ecg_peaks(
                ecg_clean,
                sampling_rate=fs,
                method=PEAK_METHOD,
                correct_artifacts=True
            )
            return np.array(info["ECG_R_Peaks"], dtype=int)
        except TypeError:
            # Phiên bản cũ không hỗ trợ correct_artifacts
            pass

    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method=PEAK_METHOD)
    rpeaks = np.array(info.get("ECG_R_Peaks", []), dtype=int)

    if CORRECT_ARTIFACTS and len(rpeaks) > 0:
        try:
            fixed_peaks = nk.signal_fixpeaks(
                rpeaks,
                sampling_rate=fs,
                method=FIXPEAKS_METHOD,
                show=False
            )
            if isinstance(fixed_peaks, tuple):
                fixed_peaks = fixed_peaks[0]
            if isinstance(fixed_peaks, dict):
                fixed_peaks = fixed_peaks.get("Peaks", fixed_peaks.get("ECG_R_Peaks"))
            return np.array(fixed_peaks, dtype=int)
        except Exception as e:
            print(f"[Warn] Fixpeaks failed ({e}), using raw peaks.")

    return rpeaks


def debug_checks(demo_signal, demo_peaks_indices, demo_peak_times, demo_peak_values, fs, stage_name="Demo Slice"):
    """In báo cáo kiểm tra logic dữ liệu (alignment, bounds, time axis)."""
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
        print(f"   ❌ [FAIL] Index lỗi! Peak index nằm ngoài phạm vi tín hiệu [0, {n_samples-1}].")
        print(f"      Min peak: {min_idx}, Max peak: {max_idx}")
    else:
        print("   ✅ [PASS] Bounds check: Tất cả đỉnh nằm gọn trong cửa sổ.")

    signal_values_at_peaks = demo_signal[demo_peaks_indices]
    diff = np.abs(signal_values_at_peaks - demo_peak_values)
    max_diff = float(np.max(diff))

    if max_diff > 1e-6:
        print("   ❌ [FAIL] LỆCH ĐỈNH! Chấm đỏ không nằm trên đường xanh.")
        print(f"      Sai số lớn nhất: {max_diff:.6f}")
    else:
        print("   ✅ [PASS] Alignment check: Chấm đỏ khớp 100% với dây tín hiệu.")

    calculated_times = demo_peaks_indices / fs
    time_diff = np.abs(calculated_times - demo_peak_times)
    if float(np.max(time_diff)) > 1e-6:
        print("   ❌ [FAIL] Lỗi trục thời gian (Time Axis).")
    else:
        print("   ✅ [PASS] Time axis check: Trục thời gian chuẩn.")

    print("--------------------------------------------------\n")


# =============================
# CHƯƠNG TRÌNH CHÍNH
# =============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_plot_style(BASE_FONT)

    # -----------------------------
    # BƯỚC 1: LOAD DỮ LI LIỆU GỐC
    # -----------------------------
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
        raise ValueError(f"lead_idx={lead_idx} out of range. Record has {n_ch} channel(s).")

    raw_ecg_full = record.p_signal[:, lead_idx]

    if hasattr(record, "sig_name") and record.sig_name is not None and len(record.sig_name) > lead_idx:
        print(f"[Info] sig_name[{lead_idx}] = {record.sig_name[lead_idx]}")

    if SKIP_SAMPLES > 0:
        if SKIP_SAMPLES >= len(raw_ecg_full):
            raise RuntimeError(f"SKIP_SAMPLES={SKIP_SAMPLES} >= signal length {len(raw_ecg_full)}.")
        raw_ecg_full = raw_ecg_full[SKIP_SAMPLES:]

    # -----------------------------
    # BƯỚC 2: CHẠY PIPELINE CHÍNH (TRÊN CỬA SỔ LỚN)
    # -----------------------------
    n_hrv = int(hrv_window_sec * fs)
    if len(raw_ecg_full) >= n_hrv:
        raw_main = raw_ecg_full[:n_hrv]
    else:
        raw_main = raw_ecg_full
        print(f"[Warn] Record shorter than {hrv_window_sec}s. Using full length ({len(raw_main)/fs:.1f}s).")

    ecg_clean_main = nk.ecg_clean(raw_main, sampling_rate=fs)
    rpeaks_main = detect_rpeaks_robust(ecg_clean_main, fs)

    if len(rpeaks_main) < 10:
        raise RuntimeError(f"Too few R-peaks detected in main window: {len(rpeaks_main)}.")

    times_r_main = rpeaks_main / fs
    rri_main = np.diff(times_r_main)
    times_rri_main = times_r_main[1:]

    # Nội suy RRI sang 4Hz
    if len(times_rri_main) < 3:
        raise RuntimeError("Not enough RR intervals for interpolation/HRV.")

    t_interp_main = np.arange(times_rri_main[0], times_rri_main[-1], 1.0 / fs_interp)

    f_interp = interp1d(
        times_rri_main,
        rri_main,
        kind="quadratic",
        fill_value="extrapolate",
        bounds_error=False,
        assume_sorted=True
    )
    rri_interp_main = f_interp(t_interp_main)

    # HRV
    rri_time_for_hrv = (t_interp_main - t_interp_main[0]).astype(float)
    try:
        hrv_results = nk.hrv(
            {"RRI": rri_interp_main * 1000.0, "RRI_Time": rri_time_for_hrv},
            sampling_rate=fs_interp,
            show=False
        )
    except Exception as e:
        print(f"❌ Error computing HRV: {e}")
        return

    # -----------------------------
    # BƯỚC 3: CẮT ĐOẠN DEMO ĐỂ VẼ (DEMO SNIPPET)
    # -----------------------------
    start_idx = int(demo_start_sec * fs)
    n_demo = int(demo_duration_sec * fs)
    end_idx = start_idx + n_demo

    if end_idx > len(ecg_clean_main):
        print("[Warn] Demo segment out of bounds. Adjusting to end of record.")
        end_idx = len(ecg_clean_main)
        start_idx = max(0, end_idx - n_demo)

    raw_demo = raw_main[start_idx:end_idx]
    clean_demo = ecg_clean_main[start_idx:end_idx]
    t_demo = np.arange(len(clean_demo)) / fs  # 0 -> demo_duration

    # Peaks within demo (index-based để khớp pixel)
    mask_peaks = (rpeaks_main >= start_idx) & (rpeaks_main < end_idx)
    rpeaks_demo_indices = (rpeaks_main[mask_peaks] - start_idx).astype(int)

    # (x, y) cho scatter
    peak_times_demo = rpeaks_demo_indices / fs
    peak_values_demo = clean_demo[rpeaks_demo_indices]

    debug_checks(
        demo_signal=clean_demo,
        demo_peaks_indices=rpeaks_demo_indices,
        demo_peak_times=peak_times_demo,
        demo_peak_values=peak_values_demo,
        fs=fs,
        stage_name="Panel C Visualization"
    )

    # RRI cắt theo thời gian demo (giây tuyệt đối của cửa sổ main)
    demo_start_t = start_idx / fs
    demo_end_t = end_idx / fs

    mask_rri = (times_rri_main >= demo_start_t) & (times_rri_main < demo_end_t)
    t_rri_demo = times_rri_main[mask_rri] - demo_start_t
    val_rri_demo = rri_main[mask_rri]

    mask_interp = (t_interp_main >= demo_start_t) & (t_interp_main < demo_end_t)
    t_int_demo = t_interp_main[mask_interp] - demo_start_t
    val_int_demo = rri_interp_main[mask_interp]

    x_max = len(clean_demo) / fs

    # -----------------------------
    # BƯỚC 4: VẼ VÀ XUẤT FILE PDF (PANELS)
    # -----------------------------

    # --- PANEL A: RAW INPUT ---
    figA, axA = plt.subplots(figsize=(6, 2.1))
    axA.plot(t_demo, raw_demo, color="#555555", linewidth=0.9)
    axA.set_xlim(0, x_max)
    axA.set_ylabel("Voltage (mV)")
    axA.set_xlabel("Time (s)")
    axA.set_title(r"$x_{\mathrm{raw}}(t)$")
    axA.grid(True)
    save_pdf(figA, os.path.join(OUT_DIR, "panel_A_raw.pdf"))

    # --- PANEL B: FILTERING ---
    figB, axB = plt.subplots(figsize=(6, 2.1))
    axB.plot(t_demo, clean_demo, color="#1f77b4", linewidth=0.9)
    axB.set_xlim(0, x_max)
    axB.set_ylabel("Voltage (mV)")
    axB.set_xlabel("Time (s)")
    axB.set_title(r"$x_{\mathrm{clean}}(t)$")
    axB.grid(True)
    save_pdf(figB, os.path.join(OUT_DIR, "panel_B_clean.pdf"))

    # --- PANEL C: FEATURE EXTRACTION (2 Subplots) ---
    figC, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 4.4),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35}
    )

    # Subplot C1: R-peak Detection
    ax1.plot(t_demo, clean_demo, color="#1f77b4", linewidth=0.9)
    ax1.scatter(peak_times_demo, peak_values_demo, color="red", s=25, zorder=5, label="R-peaks")
    ax1.set_xlim(0, x_max)
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("R-peak Detection")
    ax1.grid(True)

    # Subplot C2: RRI Time Series
    ax2.step(t_rri_demo, val_rri_demo, where="pre", color="gray", alpha=0.6, label="Raw RRI", linewidth=1.2)
    ax2.plot(t_int_demo, val_int_demo, color="#d62728", linewidth=1.5, label="4 Hz Interpolated")
    ax2.set_xlim(0, x_max)
    ax2.set_ylabel("RRI (s)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    # ✅ Legend ngoài vùng axes (margin phải) - 1 legend chung cho cả figure
    figC.subplots_adjust(right=0.78)  # chừa lề phải
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

    # Mũi tên nối giữa 2 subplot
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

    # --- PANEL HRV TABLE ---
    cols_wanted = ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_MeanNN", "HRV_LFHF"]
    available_cols = [c for c in cols_wanted if c in hrv_results.columns]

    if available_cols:
        row_hrv = hrv_results.iloc[0][available_cols]

        table_data = []
        for col in cols_wanted:
            if col in row_hrv:
                val = row_hrv[col]
                key_nice = col.replace("HRV_", "")
                if key_nice in ["SDNN", "RMSSD", "MeanNN"]:
                    val_str = f"{float(val):.1f}"
                elif key_nice == "pNN50":
                    val_str = f"{float(val):.1f}"
                else:
                    val_str = f"{float(val):.2f}"
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
    else:
        print("⚠️ Warning: No HRV features computed to generate table.")

    print("\n✅ DONE! All PDF panels saved in 'pdf/' folder.")


if __name__ == "__main__":
    main()
