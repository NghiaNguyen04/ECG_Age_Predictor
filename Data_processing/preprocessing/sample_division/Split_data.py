"""
Hàm này tạo csv chứa cột [ID, Age_group, duration_sec, split]
với 'split' có 2 giá trị [train, test]

* Qui tắc chi sao cho cho duration_sec của từng Age_group của tập
train và test có tỷ lệ giống nhau và tập test ko có subject từ tập train (ko lộ bệnh nhân)

Kết quả:

=== TỔNG QUAN (giây) ===
Tổng thời lượng  : 1,182,401.7
Train (~70.00%): 827,669.9
Test  (~30.00%): 354,731.7  (sai lệch 11.2s)

=== THEO Age_group ===
 Age_group  total_sec  target_test_sec  got_test_sec  n_id_total  n_id_test
       0.0 747942.032      224382.6096    224384.511         697        209
       1.0 166169.649       49850.8947     49857.576         148         45
       2.0 116567.273       34970.1819     34970.604         100         28
       3.0 151722.698       45516.8094     45519.041         150         44

=== PHÂN BỐ LỚP (ID-level) ===
                 n_id         sec
split Age_group
test  0.0         209  224384.511
      1.0          45   49857.576
      2.0          28   34970.604
      3.0          44   45519.041
train 0.0         488  523557.521=
      1.0         103  116312.073
      2.0          72   81596.669
      3.0         106  106203.657

"""


import os, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
import argparse

RNG = np.random.RandomState

def find_records(root_dir):
    """Tìm tất cả bản ghi theo cặp .hea/.dat; trả về danh sách path KHÔNG phần mở rộng."""
    heas = glob.glob(os.path.join(root_dir, "**", "*.hea"), recursive=True)
    records = []
    for hea in heas:
        base = os.path.splitext(hea)[0]
        dat = base + ".dat"
        if os.path.exists(dat):
            records.append(base)
    return sorted(records)

def read_duration_from_header(record_base):
    """Đọc thời lượng (giây) từ header bằng wfdb mà không cần load tín hiệu."""
    # wfdb expects record name and directory separately
    p = Path(record_base)
    header = wfdb.rdheader(p)
    fs = float(header.fs)
    sig_len = int(header.sig_len)
    return sig_len / fs

def default_parse_id_from_name(filename, id_regex=r"(\d+)"):
    """Mặc định: lấy ID là cụm số đầu tiên trong tên file (không phần mở rộng)."""
    m = re.search(id_regex, filename)
    if not m:
        raise ValueError(f"Không tách được ID từ tên file: {filename}. Hãy cung cấp --id-regex phù hợp hoặc file ánh xạ.")
    return m.group(1)

def build_records_table(root_dir, subject_info_csv, id_regex=None, map_csv=None):
    """
    Tạo bảng records: mỗi dòng là một bản ghi (record), gồm:
    record_base, record_name, path, ID, Age_group, fs, sig_len, duration_sec
    """
    # 1) Load subject-info
    subj = pd.read_csv(subject_info_csv, dtype={'ID': str})
    assert {"ID", "Age_group"}.issubset(subj.columns), "subject-info.csv phải có cột ID và Age_group"
    subj["ID"] = subj["ID"].astype(str)
    subj = subj[["ID", "Age_group"]]

    # 2) Duyệt tất cả records
    rows = []
    records = find_records(root_dir)
    for base in tqdm(records, desc="Đọc header"):
        duration = read_duration_from_header(base)
        record_name = Path(base).name  # không phần mở rộng
        rows.append({
            "ID": record_name,
            "duration_sec": float(duration),
        })

    df_rec = pd.DataFrame(rows)
    # 3) Gắn Age_group theo ID
    df_rec["ID"] = df_rec["ID"].astype(str)
    df = df_rec.merge(subj[["ID", "Age_group"]], on="ID", how="left")
    df["duration_sec"] = df["duration_sec"].astype(np.float64)

    df.dropna(inplace=True)

    return df

def aggregate_by_id(df_records):
    """Gộp thời lượng theo ID & Age_group."""
    grp = (df_records
           .groupby(["ID", "Age_group"], as_index=False)["duration_sec"]
           .sum())
    grp["duration_sec"] = grp["duration_sec"].astype(np.float64)
    return grp

def _choose_by_duration(ids_and_durs, target, seed=42, n_trials=256):
    """
    Chọn tập con ID sao cho tổng duration gần nhất target bằng greedy nhiều lần.
    ids_and_durs: list[(id, dur)]
    Trả về: set_id_chon
    """
    rng = RNG(seed)
    best_sel, best_err, best_sum = None, float("inf"), None
    ids = [x[0] for x in ids_and_durs]
    durs = [x[1] for x in ids_and_durs]
    idxs = list(range(len(ids)))

    for t in range(n_trials):
        rng.shuffle(idxs)
        s = 0.0
        sel = []
        # Greedy: thêm dần cho tới khi vượt hoặc sát mục tiêu
        for i in idxs:
            if s < target:
                sel.append(i)
                s += durs[i]
        err = abs(s - target)
        if err < best_err:
            best_err = err
            best_sel = set(sel)
            best_sum = s
            if best_err == 0:
                break
    chosen_ids = {ids[i] for i in best_sel}
    return chosen_ids, best_sum, best_err

def stratified_time_split(df_id, test_size=0.3, seed=42, n_trials=256, verbose=True):
    """
    Chia test theo ID sao cho:
      - test chiếm ~test_size *thời lượng*
      - stratified theo Age_group (mỗi nhóm đạt ~test_size thời lượng)
    Trả về: test_ids (set), train_ids (set), thống kê.
    """
    test_ids = set()
    per_group_stats = []

    for ag, sub in df_id.groupby("Age_group"):
        total = sub["duration_sec"].sum()
        target = total * test_size
        ids_and_durs = list(zip(sub["ID"].tolist(), sub["duration_sec"].tolist()))
        chosen, got, err = _choose_by_duration(ids_and_durs, target, seed=seed+hash(ag)%10000, n_trials=n_trials)
        # Đảm bảo không chọn hết hoặc không chọn gì (giữ tối thiểu 1 ID mỗi phía nếu có thể)
        if 0 < len(chosen) < len(sub):
            pass
        else:
            # Trường hợp cực đoan: chọn ID dài nhất làm test để không rỗng
            sorted_ids = sorted(ids_and_durs, key=lambda x: x[1], reverse=True)
            chosen = {sorted_ids[0][0]}

        test_ids.update(chosen)
        per_group_stats.append({
            "Age_group": ag,
            "total_sec": float(total),
            "target_test_sec": float(target),
            "got_test_sec": float(df_id[df_id["ID"].isin(chosen)]["duration_sec"].sum()),
            "n_id_total": int(len(sub)),
            "n_id_test": int(len(chosen)),
        })

    all_ids = set(df_id["ID"].tolist())
    train_ids = all_ids - test_ids

    # Tổng hợp thống kê
    total_all = df_id["duration_sec"].sum()
    test_all = df_id[df_id["ID"].isin(test_ids)]["duration_sec"].sum()
    train_all = total_all - test_all

    if verbose:
        print("\n=== TỔNG QUAN (giây) ===")
        print(f"Tổng thời lượng  : {total_all:,.1f}")
        print(f"Train (~{(train_all/total_all)*100:.2f}%): {train_all:,.1f}")
        print(f"Test  (~{(test_all/total_all)*100:.2f}%): {test_all:,.1f}  (sai lệch {abs(test_all - total_all*test_size):,.1f}s)")
        print("\n=== THEO Age_group ===")
        df_stats = pd.DataFrame(per_group_stats)
        print(df_stats.to_string(index=False))

    return test_ids, train_ids, pd.DataFrame(per_group_stats)

def summarize_split(df_records, test_ids, out_dir):
    """Tạo các file output và in tóm tắt phân bố lớp & thời lượng."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Bảng theo ID
    df_id = aggregate_by_id(df_records)
    df_id["split"] = np.where(df_id["ID"].isin(test_ids), "test", "train")
    df_id.to_csv(os.path.join(out_dir, "split_assignments.csv"), index=False)

    # Tách thành 2 csv train/test
    train_ids = df_id[df_id['split'] == 'train']
    train_ids = train_ids[["ID", "Age_group"]]
    test_ids = df_id[df_id['split'] == 'test']
    test_ids = test_ids[["ID", "Age_group"]]

    # In thống kê nhanh
    print("\n=== PHÂN BỐ LỚP (ID-level) ===")
    print(df_id.groupby(["split","Age_group"])["duration_sec"].agg(["count","sum"]).rename(columns={"count":"n_id","sum":"sec"}))

    print("\nFile đã lưu:")
    print(f"- {os.path.join(out_dir, 'split_assignments.csv')}")

    # Lưu DataFrame train vào file 'train_ID_Group.csv'
    train_ids.to_csv(os.path.join(out_dir, "train_ID_Group.csv"), index=False)

    # Lưu DataFrame test vào file 'test_ID_Group.csv'
    test_ids.to_csv(os.path.join(out_dir, "test_ID_Group.csv"), index=False)


# ------------------------------------------------------------------------------------------
def main(
        root_dir,
        subject_info_csv,
        out_dir="./splits",
        test_size=0.30,
        seed=42,
        n_trials=512
):
    df_records = build_records_table(
        root_dir=root_dir,
        subject_info_csv=subject_info_csv,
    )

    # Gộp theo ID và split theo thời lượng + stratified theo Age_group
    df_id = aggregate_by_id(df_records)
    test_ids, train_ids, per_group = stratified_time_split(
        df_id, test_size=float(test_size), seed=int(seed), n_trials=int(n_trials), verbose=True
    )
    summarize_split(df_records, test_ids, out_dir)

# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT_DIR = '../../../data/raw/autonomic-aging-a-dataset'
    OUT_DIR = ''

    # SUBJECT_INFO_FILE = 'Data/raw/autonomic-aging-a-dataset_1D/subject-info.csv'
    # main(root_dir= ROOT_DIR,
    #      subject_info_csv=SUBJECT_INFO_FILE,
    #      out_dir=OUT_DIR)

    SUBJECT_INFO_FILE_REDUCED = '../../Data/LuuTru/Age_group_reduced.csv'

    main(root_dir= ROOT_DIR,
         subject_info_csv=SUBJECT_INFO_FILE_REDUCED,
         out_dir=OUT_DIR)
