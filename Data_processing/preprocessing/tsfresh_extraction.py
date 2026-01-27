# -*- coding: utf-8 -*-
import os, glob
os.environ["NUMBA_DISABLE_CUDA"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy.interpolate import interp1d

from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute


# ---------------------------
# 1) Đọc WFDB & chọn kênh
# ---------------------------
PREFERRED_LEADS = ['ECG1', 'ECG2']

def read_wfdb_record(basename: str):
    """
    basename: đường dẫn KHÔNG đuôi (vd: '/path/100' cho 100.dat, 100.hea)
    return: (signals: np.ndarray[nsamples, nch], fs: float, lead_names: list[str])
    """
    rec = wfdb.rdrecord(basename, channels=None, sampto=None)  # đọc full
    sig = rec.p_signal.astype(np.float64)  # (N, C)
    fs = float(rec.fs)
    leads = list(rec.sig_name)
    return sig, fs, leads

def pick_lead(sig: np.ndarray, leads: list[str]):
    """Ưu tiên chọn 1 kênh ECG 'đẹp' theo danh sách PREFERRED_LEADS; nếu không có thì lấy kênh 0."""
    if sig.ndim == 1:
        return sig, "ECG1"
    if len(leads) != sig.shape[1]:
        # phòng trường hợp thiếu tên kênh
        idx = 0
    else:
        idx = 0
        for name in PREFERRED_LEADS:
            if name in leads:
                idx = leads.index(name)
                break
    return sig[:, idx].copy(), (leads[idx] if idx < len(leads) else "ECG1")


# --------------------------------------
# 2) Làm sạch + nội suy về target_fs
# --------------------------------------
def clean_and_resample(ecg: np.ndarray, fs: int, target_fs: int = 250, method: str = "linear"):
    """
    - Làm sạch ECG bằng neurokit2.ecg_clean
    - Nội suy (linear/cubic) về target_fs bằng scipy.interpolate.interp1d
    """
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs, method="neurokit")  # hoặc "biosppy","elgendi"
    if abs(target_fs - fs) < 1e-9:
        return ecg_clean, fs

    n = ecg_clean.shape[0]
    t_old = np.arange(n, dtype=np.float64) / fs
    n_new = int(round(n * target_fs / fs))
    t_new = np.arange(n_new, dtype=np.float64) / target_fs

    # kind: "linear" nhanh, "cubic" mượt hơn nhưng nặng hơn
    f = interp1d(t_old, ecg_clean.astype(np.float64), kind=method, bounds_error=False, fill_value="extrapolate")
    ecg_rs = f(t_new).astype(np.float64)
    return ecg_rs, float(target_fs)


# ---------------------------
# 3) Chia segment trượt
# ---------------------------
def sliding_windows(sig: np.ndarray, fs: float, window_sec: float = 10.0, step_sec: float = 5.0):
    """
    Trả về list các đoạn (mỗi đoạn độ dài = window_sec).
    """
    L = len(sig)
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 0 or step <= 0 or L < win:
        return []

    windows = []
    starts = range(0, L - win + 1, step)
    for s in starts:
        windows.append(sig[s:s+win])
    return windows


# -------------------------------------------------
# 5) Ghép DataFrame long-format cho tsfresh
# -------------------------------------------------
def segments_to_long_df(segments, fs: float, rec_name: str, lead_name: str):
    """
    Mỗi segment -> 1 id riêng: f"{rec_name}__{lead_name}__{i}"
    Trả về DataFrame cột: id, time, value, kind (tuỳ chọn)
    """
    rows = []
    for i, seg in enumerate(segments):
        seg = np.asarray(seg, dtype=np.float64)
        t = np.arange(seg.shape[0], dtype=np.float64) / fs
        seg_id = f"{rec_name}__{lead_name}__{i:05d}"
        rows += [{"id": seg_id, "time": float(tk), "value": float(v)} for tk, v in zip(t, seg)]
    return pd.DataFrame(rows)


# -------------------------------------------------
# 6) End-to-end: từ thư mục WFDB → DataFrame cho tsfresh
# -------------------------------------------------
def build_tsfresh_container_from_wfdb(
        root_dir: str,
        target_fs: int = 4.0,
        window_sec: float = 10.0,
        step_sec: float = 5.0,
        use_quality_filter: bool = True,
        interp_method: str = "linear",
):
    """
    - Dò toàn bộ *.hea trong root_dir, đọc cặp .dat/.hea
    - Chọn 1 lead  -> làm sạch -> nội suy -> chia segment -> (lọc chất lượng) -> gộp long-format
    """
    root = Path(root_dir)
    df_all = []
    meta = []  # lưu thông tin để map ngược (id_segment → record, lead)
    for hea in glob.glob(str(root / "*.hea")):
        base = os.path.splitext(hea)[0]
        rec_name = os.path.basename(base)

        try:
            sig, fs, leads = read_wfdb_record(base)
        except Exception as e:
            print(f"[WARN] Bỏ {rec_name}: lỗi đọc WFDB ({e})")
            continue

        # ecg_1ch, lead_name = pick_lead(sig, leads)
        ecg_1ch, lead_name = sig[:, 0], "ECG1" # lấy kênh đầu tiên

        # clean + resample
        ecg_rs, fs_rs = clean_and_resample(ecg_1ch, int(fs), target_fs=target_fs, method=interp_method)

        # segment
        segs = sliding_windows(ecg_rs, fs_rs, window_sec=window_sec, step_sec=step_sec)

        if len(segs) == 0:
            continue

        df_rec = segments_to_long_df(segs, fs_rs, rec_name, lead_name)
        df_all.append(df_rec)

        # meta để tra cứu
        for i in range(len(segs)):
            seg_id = f"{rec_name}__{lead_name}__{i:05d}"
            meta.append({"id": seg_id, "record": rec_name, "lead": lead_name, "fs": fs_rs})

    if len(df_all) == 0:
        raise RuntimeError("Không thu được segment nào sau lọc.")

    df_long = pd.concat(df_all, ignore_index=True)
    meta_df = pd.DataFrame(meta)
    # đảm bảo dtype
    df_long["id"] = df_long["id"].astype(str)
    df_long["time"] = df_long["time"].astype(np.float64)
    df_long["value"] = df_long["value"].astype(np.float64)

    return df_long, meta_df


# -------------------------------------------------
# 7) Ví dụ sử dụng + trích đặc trưng với tsfresh
# -------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "../../data/raw/autonomic-aging-a-dataset"  # chứa các cặp .dat .hea
    OUT_DIR = "../../data/interim"
    SUBJECT_INFO_csv = "../data/processed/subject_reduced.csv"
    ID_LOW_QUALITY_csv = "../eda/eda_raw_data/ecg_quality_Simple.csv"

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    df_long, meta_df = build_tsfresh_container_from_wfdb(
        ROOT_DIR,
        target_fs=4,            #
        window_sec=300.0,       # mỗi segment 300s
        step_sec=300.0,         # non-overlap
        use_quality_filter=True,
        interp_method="quadratic"
    )

    print(df_long.head())
    print(meta_df.head())

    # 1) Trích đặc trưng cho từng segment-id
    X = extract_features(
        df_long,
        column_id="id",           # <-- phải là 'id'
        column_sort="time",
        column_value="value",
        default_fc_parameters=ComprehensiveFCParameters(),
        n_jobs=0,
        disable_progressbar=True
    )

    # 2) Xử lý thiếu/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    impute(X)

    # 3) Đọc file nhãn: Age_group_reduced.csv (ID + Age_group_reduced)
    labels_path = Path(OUT_DIR) / "Age_group_reduced.csv"
    labels_df = pd.read_csv(labels_path)

    # Đảm bảo kiểu cho key join (nếu cần)
    labels_df["ID"] = labels_df["ID"].astype(str)
    meta_df["record"] = meta_df["record"].astype(str)

    # meta_df: ['id','record','lead','fs']
    # Join để gán Age_group_reduced cho từng segment-id
    meta_with_y = meta_df.merge(
        labels_df,
        left_on="record",
        right_on="ID",
        how="left"
    )

    # 4) Kiểm tra thiếu nhãn
    if meta_with_y["Age_group_reduced"].isna().any():
        n_missing = meta_with_y["Age_group_reduced"].isna().sum()
        print(f"[WARN] Có {n_missing} segment không có Age_group_reduced, sẽ bị loại.")
        meta_with_y = meta_with_y.dropna(subset=["Age_group_reduced"])

    # 5) Tạo Series y với index = segment-id (id), trùng index của X
    y = (
        meta_with_y
        .set_index("id")["Age_group_reduced"]
        .astype(int)
    )

    # Đồng bộ X và y: chỉ giữ các id chung
    common_ids = X.index.intersection(y.index)
    X = X.loc[common_ids].copy()
    y = y.loc[common_ids].copy()

    print("Shape trước chọn feature:", X.shape)

    # 6) Chọn feature quan trọng
    X_selected = select_features(X, y)

    print("Shape sau chọn feature:", X_selected.shape)

    # 7) Lưu cả full và selected
    X_full_out = X.copy()
    X_full_out.insert(0, "id", X_full_out.index)   # đây là segment-id
    X_full_out.to_csv(Path(OUT_DIR) / "features_tsfresh_full.csv", index=False)

    X_sel_out = X_selected.copy()
    X_sel_out.insert(0, "id", X_sel_out.index)
    X_sel_out.to_csv(Path(OUT_DIR) / "features_tsfresh_selected.csv", index=False)


#%%
