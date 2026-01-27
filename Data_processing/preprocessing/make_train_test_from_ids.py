#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import re
import pandas as pd
from pathlib import Path

ID_PAT = re.compile(r'^id(?:\.\d+)?$', re.IGNORECASE)

def _collapse_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Find columns named ID, ID.1, ID.2, ... (case-insensitive)
    id_like = [c for c in df.columns if ID_PAT.match(c.strip())]
    if not id_like:
        return df
    # Ensure all are string for safe combine
    for c in id_like:
        df[c] = df[c].astype('string').str.strip()
    # Combine-first across the ID-likes in order of appearance
    base = id_like[0]
    combined = df[base].copy()
    for c in id_like[1:]:
        combined = combined.combine_first(df[c])
    df['ID'] = combined
    # Drop all extra ID-like columns except the final 'ID'
    drop_cols = [c for c in id_like if c != 'ID']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    # Ensure 'ID' is the first column
    cols = ['ID'] + [c for c in df.columns if c != 'ID']
    df = df[cols]
    return df

def read_csv_force_id_str(path: Path):
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    # Collapse duplicate ID columns
    df = _collapse_id_columns(df)
    if 'ID' not in df.columns:
        print(f"[ERROR] {path} is missing 'ID' after normalization.", file=sys.stderr)
        sys.exit(1)
    # Final tidy
    df['ID'] = df['ID'].astype('string').str.strip()
    return df

def ensure_unique_ids(df: pd.DataFrame, name: str):
    dup = df['ID'][df['ID'].duplicated(keep=False)]
    if not dup.empty:
        print(f"[WARN] {name} has duplicated IDs ({dup.nunique()} unique). Keeping first occurrence.", file=sys.stderr)
        df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    return df

def coerce_age_col(df: pd.DataFrame, col: str):
    if col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise').astype('Int64')
        except Exception:
            pass  # leave as-is

def resolve_label_after_merge(merged: pd.DataFrame, age_col: str, prefer_source: str):
    if age_col in merged.columns:
        return merged
    left = f"{age_col}_x"
    right = f"{age_col}_y"
    if left in merged.columns and right in merged.columns:
        both_mask = merged[left].notna() & merged[right].notna()
        conflicts = (merged.loc[both_mask, left] != merged.loc[both_mask, right]).sum()
        if conflicts:
            print(f"[WARN] Detected {conflicts} conflicting labels; keeping {prefer_source}.", file=sys.stderr)
        if prefer_source == 'train_test':
            merged[age_col] = merged[left].combine_first(merged[right])
        else:
            merged[age_col] = merged[right].combine_first(merged[left])
        merged.drop(columns=[left, right], inplace=True)
    elif left in merged.columns:
        merged.rename(columns={left: age_col}, inplace=True)
    elif right in merged.columns:
        merged.rename(columns={right: age_col}, inplace=True)
    else:
        merged[age_col] = pd.Series([pd.NA] * len(merged), dtype='Int64')
        print(f"[WARN] No '{age_col}' found in either train/test or data; filled with NA.", file=sys.stderr)
    return merged

def main():
    ap = argparse.ArgumentParser(description='Join master data with train/test ID lists; collapse duplicate ID columns.')
    ap.add_argument('--data', type=Path, required=True, help="Master CSV (must contain 'ID' or ID.1 etc.; may contain 'Age_group')")
    ap.add_argument('--train', type=Path, required=True, help="Train CSV (must contain 'ID' or ID.1 etc.; may contain 'Age_group')")
    ap.add_argument('--test', type=Path, required=True, help="Test CSV (must contain 'ID' or ID.1 etc.; may contain 'Age_group')")
    ap.add_argument('--out_dir', type=Path, default=Path('.'), help='Output directory')
    ap.add_argument('--age_col', type=str, default='Age_group', help='Label column name to use/resolve')
    ap.add_argument('--prefer_label', type=str, choices=['train_test','data'], default='train_test', help='If both sides have labels, which to keep')
    args = ap.parse_args()

    df_data  = read_csv_force_id_str(args.data)
    df_train = read_csv_force_id_str(args.train)
    df_test  = read_csv_force_id_str(args.test)

    coerce_age_col(df_data,  args.age_col)
    coerce_age_col(df_train, args.age_col)
    coerce_age_col(df_test,  args.age_col)

    df_data  = ensure_unique_ids(df_data,  'data')
    df_train = ensure_unique_ids(df_train, 'train')
    df_test  = ensure_unique_ids(df_test,  'test')

    train_joined = df_train.merge(df_data, on='ID', how='left')
    test_joined  = df_test.merge(df_data,  on='ID', how='left')

    train_joined = resolve_label_after_merge(train_joined, args.age_col, args.prefer_label)
    test_joined  = resolve_label_after_merge(test_joined,  args.age_col, args.prefer_label)

    # Reorder: ID, Age_group, then features
    def reorder(df):
        cols = list(df.columns)
        cols = [c for c in cols if c != 'ID']
        if args.age_col in cols:
            cols.remove(args.age_col)
        return df[['ID', args.age_col] + cols]

    train_joined = reorder(train_joined)
    test_joined  = reorder(test_joined)

    # Cast numeric-like columns
    for df in (train_joined, test_joined):
        for c in df.select_dtypes(include=['float', 'float16', 'float32']).columns:
            df[c] = df[c].astype('float64')
        for c in df.select_dtypes(include=['int8','int16','int32']).columns:
            df[c] = df[c].astype('int64')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_train = args.out_dir / 'train_data.csv'
    out_test  = args.out_dir / 'test_data.csv'
    train_joined.to_csv(out_train, index=False)
    test_joined.to_csv(out_test,  index=False)

    print('[OK] Saved:')
    print(f' - {out_train}  (rows={{len(train_joined)}}, cols={{train_joined.shape[1]}})')
    print(f' - {out_test}   (rows={{len(test_joined)}},  cols={{test_joined.shape[1]}})')

if __name__ == '__main__':
    main()


# python preprocessing/make_train_test_from_ids.py `
# --data "data/interim/data_60s/rri_hrv_df.csv" `
# --train "data/interim/split_train_test/train_ID_Group.csv" `
# --test "data/interim/split_train_test/test_ID_Group.csv" `
# --out_dir "data/processed/dataset_1D/RRI_HRV_60s"