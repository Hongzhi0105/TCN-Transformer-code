# datasetdiff.py
# Shifted Mode:
# X = [past(t-Input~t-1) using X_PAST_COLS] + [future(t0~t+N-1) using X_FUTURE_COLS]
# y = obs_diff(t0~t+N-1)
# Cliff prevention: past-only cols in future segment filled by last past value (t-1)

import argparse
import json
import numpy as np
import pandas as pd
import calendar
import torch
from torch.utils.data import TensorDataset

import origin_config as config


def arg():
    parser = argparse.ArgumentParser(description="Dataset Generation (Shifted Mode)")
    parser.add_argument('--Input_timestep', type=int, default=config.Input_timesteps, help='Input_timestep (past length).')
    parser.add_argument('--Output_timestep', type=int, default=config.Output_timesteps, help='Output_timestep (future length).')
    parser.add_argument('--site_sets', type=str, default=str(config.site_sets), help='site_sets, e.g. "[68]".')
    parser.add_argument('--output_dir', type=str, default='results', help='Output folder name.')
    args = parser.parse_args()

    with open(f"./{args.output_dir}/Arglog.txt", 'w') as f:
        f.write(f"Dataset Generation (Shifted Mode): Input={args.Input_timestep}, Output={args.Output_timestep}\n")
        f.write(f"X_PAST_COLS={config.X_PAST_COLS}\n")
        f.write(f"X_FUTURE_COLS={config.X_FUTURE_COLS}\n")
        f.write(f"X_UNION_COLS={config.X_UNION_COLS}\n")
    return args


def data_gen(site_sets, time_steps, output_timesteps, output_dir):
    OrigData_path = config.OrigData_path
    _fileName_sets = config._fileName_sets

    HIST_DIR = getattr(config, "HIST_DIR", "./HIST_DIR")
    year_of_data = getattr(config, "year_of_data", 2016)

    X_PAST_COLS = config.X_PAST_COLS
    X_FUTURE_COLS = config.X_FUTURE_COLS
    X_UNION_COLS = config.X_UNION_COLS
    Y_COL = config.Y_COL

    need_x = set(config.Need_element_X)
    need_y = set(config.Need_element_Y)
    needed = need_x.union(need_y)

    input_dim = len(X_UNION_COLS)
    total_seq_len = time_steps + output_timesteps

    # past-only cols: present in past but not in future
    past_only_cols = [c for c in X_PAST_COLS if c not in X_FUTURE_COLS]

    # indices in union space
    idx_past = [X_UNION_COLS.index(c) for c in X_PAST_COLS]
    idx_future = [X_UNION_COLS.index(c) for c in X_FUTURE_COLS]
    idx_past_only = [X_UNION_COLS.index(c) for c in past_only_cols]

    for _s in site_sets:
        _dict = {}

        # ---- 1) read full year (mon 1..12) ----
        for _base in _fileName_sets:
            for mon_i in range(1, 13):
                format_mon = str(mon_i).zfill(2)
                filename = list(_base)
                filename.insert(4, format_mon)
                filename = ''.join(filename)

                try:
                    with open(f'{OrigData_path}/{filename}.txt', 'r') as f:
                        for line in f:
                            _temp = line.strip().split()
                            if len(_temp) < 5:
                                continue
                            site_id = int(_temp[0])
                            cls = _temp[2]

                            if site_id == int(_s) and (cls in needed):
                                aft_4 = _temp[4:]
                                vals = [float(v) for v in aft_4]

                                if cls not in _dict:
                                    _dict[cls] = vals
                                else:
                                    _dict[cls] += vals
                except FileNotFoundError:
                    continue

        pd_data = pd.DataFrame(_dict)

        # ---- 2) clean & interpolate ----

        # 2-1) 缺值標記：-99.9 -> NaN
        pd_data.replace(-99.9, np.nan, inplace=True)
        # 2-2) 其他欄位照原本 linear interpolate（避開 PM2.5_obs，避免長段被先補掉）
        other_cols = [c for c in pd_data.columns if c != "PM2.5_obs"]
        if other_cols:
            pd_data[other_cols] = pd_data[other_cols].interpolate(method="linear", inplace=False)
        # 2-3) PM2.5_obs 的分段補值
        if "PM2.5_obs" in pd_data.columns:
            s = pd_data["PM2.5_obs"].copy()
            # (a) 先補「短洞」：最多連續 2 筆 NaN 才會被內插補上
            #     limit_area='inside'：不補頭尾，避免沒有前/後值造成不合理外插
            s = s.interpolate(method="linear", limit=2, limit_area="inside")
            
            # (b) 若仍有 NaN（代表長段缺值或頭尾缺值），用 HIST 進行同月日小時平均補值
            if s.isna().any():
                days_in_year = 366 if calendar.isleap(year_of_data) else 365
                expected_hours = days_in_year * 24

                if len(s) != expected_hours:
                    raise ValueError(
                            f"[Site {_s}] 資料長度 ({len(s)}) 與 year_of_data={year_of_data} 的全年小時數 ({expected_hours}) 不一致，"
                            f"無法安全進行 HIST 補值。"
                    )

                time_axis = pd.date_range(
                        start=f"{year_of_data}-01-01 00:00:00",
                        periods=len(s),
                        freq="h"
                )

                hist_path = f"{HIST_DIR}/station{int(_s)}.xlsx"
                try:
                    hist_df = pd.read_excel(hist_path)
                except Exception as e:
                    raise ValueError(f"[Site {_s}] 讀取 HIST 失敗: {hist_path} ({e})")

                if hist_df.shape[1] < 2:
                    raise ValueError(f"[Site {_s}] HIST 格式不正確（至少要有 time + 年份欄位）: {hist_path}")

                time_col = hist_df.columns[0]
                year_cols = list(hist_df.columns[1:])  # 例如 2013/2014/2015（可能是 int 或 str）

                hist_df[time_col] = pd.to_datetime(hist_df[time_col], errors="coerce")
                if hist_df[time_col].isna().any():
                    raise ValueError(f"[Site {_s}] HIST time 欄位存在無法解析的時間值: {hist_path}")

                # key = (month, day, hour)
                hist_key = list(zip(
                    hist_df[time_col].dt.month,
                    hist_df[time_col].dt.day,
                    hist_df[time_col].dt.hour
                ))

                hist_vals = hist_df[year_cols].apply(pd.to_numeric, errors="coerce")
                hist_mean = hist_vals.mean(axis=1, skipna=True)  # 可能 NaN（表示三年都 NaN）
                hist_map = dict(zip(hist_key, hist_mean.values))

                # 逐點補值：只補 s 仍為 NaN 的位置
                nan_idx = np.flatnonzero(s.isna().values)
                for idx in nan_idx:
                    dt = time_axis[idx]
                    key = (dt.month, dt.day, dt.hour)
                    v = hist_map.get(key, np.nan)
                    if pd.notna(v):
                        s.iat[idx] = float(v)

            # (c) HIST 後若仍 NaN（最糟情況），最後用線性內插補
            if s.isna().any():
                s = s.interpolate(method="linear", limit_area="inside")

            pd_data["PM2.5_obs"] = s


        # ---- 3) build y target obs_diff ----
        if 'PM2.5_obs' not in pd_data.columns or 'PM2.5' not in pd_data.columns:
            raise ValueError("pd_data 缺少 'PM2.5_obs' 或 'PM2.5'，無法計算 obs_diff。")

        pd_data[Y_COL] = pd_data['PM2.5_obs'].values - pd_data['PM2.5'].values

        # ---- 4) sanity check columns ----
        missing_past = [c for c in X_PAST_COLS if c not in pd_data.columns]
        missing_future = [c for c in X_FUTURE_COLS if c not in pd_data.columns]
        if missing_past or missing_future:
            raise ValueError(
                f"資料缺欄位：missing_past={missing_past}, missing_future={missing_future}。"
                f"請確認 txt 是否包含這些 cls，或調整 X_PAST_COLS/X_FUTURE_COLS。"
            )

        X_list, y_list = [], []

        # ---- 5) make samples (t0 = i) ----
        # need i >= time_steps, and i+N <= len(pd_data)
        for i in range(time_steps, len(pd_data) - output_timesteps + 1):
            X_combined = np.zeros((total_seq_len, input_dim), dtype=np.float32)

            # A) past segment: t-72 ~ t-1
            past_vals = pd_data.iloc[i - time_steps:i][X_PAST_COLS].values  # (time_steps, len(X_PAST_COLS))
            X_combined[0:time_steps, idx_past] = past_vals

            # B) future segment: t0 ~ t+N-1
            future_vals = pd_data.iloc[i:i + output_timesteps][X_FUTURE_COLS].values  # (N, len(X_FUTURE_COLS))
            X_combined[time_steps:, idx_future] = future_vals

            # C) cliff prevention: fill past-only cols in future segment with last past value (t-1)
            if idx_past_only:
                last_past = pd_data.iloc[i - 1][past_only_cols].values.astype(np.float32)  # (len(past_only_cols),)
                X_combined[time_steps:, idx_past_only] = last_past  # broadcast to (N, len(past_only_cols))

            # D) target y: obs_diff(t0~t+N-1)
            y = pd_data.iloc[i:i + output_timesteps][[Y_COL]].values.astype(np.float32)  # (N, 1)

            X_list.append(X_combined)
            y_list.append(y)

        X_array = np.asarray(X_list, dtype=np.float32)
        y_array = np.asarray(y_list, dtype=np.float32)

        # shape checks
        if X_array.ndim != 3 or X_array.shape[2] != config.INPUT_DIM:
            raise ValueError(
                f"X shape mismatch: got {X_array.shape}, expected (*, {time_steps + output_timesteps}, {config.INPUT_DIM}). "
                f"請檢查 X_UNION_COLS 與 INPUT_DIM。"
            )
        if y_array.ndim != 3 or y_array.shape[1] != output_timesteps or y_array.shape[2] != 1:
            raise ValueError(
                f"y shape mismatch: got {y_array.shape}, expected (*, {output_timesteps}, 1)."
            )

        print(f"[Site {_s}] X shape: {X_array.shape}  y shape: {y_array.shape}")

        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        y_tensor = torch.tensor(y_array, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)

        save_path = f"./{output_dir}/{_s}_diff_training_data.pt"
        torch.save(dataset, save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    args = arg()
    data_gen(
        site_sets=json.loads(args.site_sets),
        time_steps=args.Input_timestep,
        output_timesteps=args.Output_timestep,
        output_dir=args.output_dir
    )
