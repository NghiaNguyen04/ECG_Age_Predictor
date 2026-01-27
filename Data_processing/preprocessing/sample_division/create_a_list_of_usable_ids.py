"""
Tên File: subject_data_processor.py

Mô tả:
    Tập lệnh này thực hiện quy trình tiền xử lý dữ liệu hồ sơ chủ thể (subject-info)
    dựa trên chất lượng của bản ghi điện tâm đồ (ECG) và phân nhóm lại độ tuổi.

Chức năng Chính:
1.  **Lọc dữ liệu:** Loại bỏ các ID chủ thể/file ECG có chất lượng được đánh dấu
    là 'Unacceptable' trong Quality_ecg.csv. Và ID thiếu file .dat
2.  **Chuẩn hóa ID:** So sánh các ID từ subject-info.csv với tập hợp các ID ECG
    đã được chấp nhận (Excellent/Barely acceptable) từ Quality_ecg.csv và các file
    thực tế trong thư mục 'data/'.
3.  **Biến đổi thuộc tính:** Gom nhóm lại cột 'Age_group' (1-15) thành 4 nhóm
    mới ('Age_group_reduced', từ 0-3) theo quy tắc sau:
        - Nhóm {1, 2, 3}  -> 0
        - Nhóm {4, 5}     -> 1
        - Nhóm {6, 7}     -> 2
        - Nhóm {8-15}     -> 3
4.  **Xuất dữ liệu:** Tạo file đầu ra mới tên 'subject_reduced.csv' chỉ chứa
    cột 'ID' (đã được lọc), 'Age_group_reduced', BMI, sex.

Các File Dữ liệu Đầu vào Cần Thiết:
    - data/ (Thư mục chứa các file ECG .hea và .dat)
    - Quality_ecg.csv
    - subject-info.csv

Phiên bản: 1.0
"""

#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


from pathlib import Path
import argparse
import pandas as pd


def normalize_id(x: object, width: int) -> str:
    s = str(x).strip()
    if s.isdigit() and len(s) < width:
        s = s.zfill(width)
    return s


def reduce_age_group(g: int):
    if pd.isna(g):
        return pd.NA
    g = int(g)
    if g in (1, 2, 3):
        return 0
    if g in (4, 5):
        return 1
    if g in (6, 7):
        return 2
    if 8 <= g <= 15:
        return 3
    return pd.NA


def load_valid_ids(data_root: Path):
    hea_ids = {p.stem for p in data_root.glob("*.hea")}
    dat_ids = {p.stem for p in data_root.glob("*.dat")}
    valid_ids = sorted(hea_ids & dat_ids)

    missing_dat = sorted(hea_ids - dat_ids)
    missing_hea = sorted(dat_ids - hea_ids)
    if missing_dat:
        print(f"[WARN] {len(missing_dat)} record thiếu .dat (ví dụ): {missing_dat[:10]}")
    if missing_hea:
        print(f"[WARN] {len(missing_hea)} record thiếu .hea (ví dụ): {missing_hea[:10]}")

    if not valid_ids:
        raise SystemExit("[ERROR] Không có record nào đủ cả .hea + .dat trong thư mục.")

    id_width = max(len(s) for s in valid_ids)  # ví dụ '0001' => 4
    return valid_ids, id_width


def load_drop_ids(quality_csv: Path, id_width: int, drop_barely: bool, quality_id_col: str | None) -> set[str]:
    q = pd.read_csv(quality_csv)

    # Tự nhận dạng cột ID trong quality (ưu tiên tham số, sau đó 'ID' hoặc 'name_ecg')
    id_col = quality_id_col or ("ID" if "ID" in q.columns else ("name_ecg" if "name_ecg" in q.columns else None))
    if id_col is None:
        raise SystemExit("[ERROR] Quality CSV không có cột ID ('ID' hoặc 'name_ecg').")

    q[id_col] = q[id_col].astype(str)
    q["ID"] = q[id_col].map(lambda x: normalize_id(x, id_width))

    if "quality" not in q.columns:
        raise SystemExit("[ERROR] Quality CSV thiếu cột 'quality'.")

    q["quality_low"] = q["quality"].astype(str).str.strip().str.lower()

    # Chỉ quan tâm 2 nhãn này
    q = q[q["quality_low"].isin(["unacceptable", "barely acceptable"])].copy()

    if drop_barely:
        drop_vals = ["unacceptable", "barely acceptable"]
    else:
        drop_vals = ["unacceptable"]

    drop_ids = set(q.loc[q["quality_low"].isin(drop_vals), "ID"])
    print(f"[INFO] Sẽ loại {len(drop_ids)} ID theo quality ({'Unacc+Barely' if drop_barely else 'Unacceptable only'}).")
    return drop_ids


def build_subject_reduced(
        data_root: Path,
        quality_csv: Path,
        subject_info_csv: Path,
        drop_barely: bool,
        quality_id_col: str | None,
) -> pd.DataFrame:

    valid_ids, id_width = load_valid_ids(data_root)

    # Quality như blacklist: loại những ID nằm trong quality theo rule
    drop_ids = load_drop_ids(quality_csv, id_width, drop_barely, quality_id_col)

    # Lấy tất cả ID hợp lệ trong thư mục rồi trừ blacklist
    final_ids = sorted(set(valid_ids) - drop_ids)
    if not final_ids:
        raise SystemExit("[ERROR] Sau khi loại theo quality, không còn ID nào hợp lệ.")

    # Join subject-info để lấy Age_group
    subj = pd.read_csv(subject_info_csv, dtype={"ID": str})
    subj["ID"] = subj["ID"].map(lambda x: normalize_id(x, id_width))
    subj_small = subj[["ID", "Age_group", "BMI", "Sex"]].drop_duplicates(subset="ID", keep="first")

    df = pd.DataFrame({"ID": final_ids})
    df = df.merge(subj_small, on="ID", how="left")

    missing_age = df["Age_group"].isna().sum()
    if missing_age:
        print(f"[WARN] {missing_age} ID thiếu Age_group trong subject-info.csv (sẽ bị loại).")

    df["Age_group"] = pd.to_numeric(df["Age_group"], errors="coerce").astype("Int64")

    # Reduce nhóm tuổi
    df["Age_group_reduced"] = df["Age_group"].map(reduce_age_group).astype("Int64")

    bad_group = df["Age_group_reduced"].isna().sum()
    if bad_group:
        print(f"[WARN] {bad_group} ID có Age_group ngoài [1..15] (sẽ bị loại).")

    df_out = df.dropna(subset=["Age_group_reduced"])[["ID", "Age_group_reduced", "BMI", "Sex"]].reset_index(drop=True)

    # Summary
    print("\n--- Summary ---")
    print(f"Records đủ .hea+.dat: {len(valid_ids)}")
    print(f"IDs bị loại theo quality: {len(drop_ids)}")
    print(f"IDs còn lại sau loại quality: {len(final_ids)}")
    print(f"IDs có Age_group hợp lệ 1..15: {len(df_out)}")

    return df_out


def parse_args():
    ap = argparse.ArgumentParser(description="Tạo subject_reduced.csv (blacklist từ Quality_ecg).")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Thư mục chứa .hea/.dat (vd: ../../data/raw/autonomic-aging-a-dataset)")
    ap.add_argument("--quality-csv", type=Path, required=True,
                    help="CSV có cột ID (hoặc name_ecg) và quality.")
    ap.add_argument("--subject-info-csv", type=Path, required=True,
                    help="CSV có cột ID, Age_group (1..15).")
    ap.add_argument("--out", type=Path, default=Path("subject_reduced.csv"),
                    help="File CSV xuất ra (mặc định: subject_reduced.csv).")
    ap.add_argument("--drop-barely", action="store_true",
                    help="Nếu đặt cờ này, sẽ loại cả 'Barely acceptable' (mặc định chỉ loại 'Unacceptable').")
    ap.add_argument("--quality-id-col", type=str, default=None,
                    help="Tên cột ID trong quality CSV nếu không phải 'ID' hoặc 'name_ecg'.")
    return ap.parse_args()


def main():
    args = parse_args()
    df_out = build_subject_reduced(
        data_root=args.data_root,
        quality_csv=args.quality_csv,
        subject_info_csv=args.subject_info_csv,
        drop_barely=args.drop_barely,
        quality_id_col=args.quality_id_col,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\n[OK] Đã lưu {len(df_out)} dòng vào: {args.out.resolve()}")


if __name__ == "__main__":
    main()

# python preprocessing/sample_division/create_a_list_of_usable_ids.py `
#  --data-root "data/raw/autonomic-aging-a-dataset" `
#  --quality-csv "eda/eda_raw_data/ecg_quality_fuzzy.csv" `
#  --subject-info-csv "data/raw/autonomic-aging-a-dataset/subject-info.csv" `
#  --out "data/processed/subject_reduced.csv"

# --drop-barely ` : Tùy chọn drop cả barely acceptable

"""
Kết quả: 

[WARN] 1 record thiếu .dat (ví dụ): ['0400']
[INFO] Sẽ loại 0 ID theo quality (Unacceptable only).
[WARN] 25 ID thiếu Age_group trong subject-info.csv (sẽ bị loại).
[WARN] 25 ID có Age_group ngoài [1..15] (sẽ bị loại).

--- Summary ---
Records đủ .hea+.dat: 1118
IDs bị loại theo quality: 0
IDs còn lại sau loại quality: 1118
IDs có Age_group hợp lệ 1..15: 1093

* Lưu ý: đã loại 2 subject: 0167 và 1011 ở data-root raw, vì 2 file này missing quá nhiều dữ liệu
"""