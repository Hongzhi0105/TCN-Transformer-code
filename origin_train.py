# train.py
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset

import origin_config as config


def arg():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--Input_timestep', type=int, default=config.Input_timesteps, help='Input_timestep.')
    parser.add_argument('--Output_timestep', type=int, default=config.Output_timesteps, help='Output_timestep.')
    parser.add_argument('--Avoid_interval', type=int, default=7, help='Avoid_interval (month 1~12, 0 means no avoid).')
    parser.add_argument('--site_sets', type=str, default=str(config.site_sets), help='site_sets.')
    parser.add_argument('--output_dir', type=str, default='results', help='Output folder name.')
    return parser.parse_args()


def train(site_sets, time_steps, output_timesteps, output_dir, train_loader, valid_loader, RatioInterval='ratio'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class TCN_block(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=(kernel_size - 1) * dilation, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_channels)
            #self.relu1 = nn.ReLU()
            #self.relu1 = nn.PReLU(out_channels)
            self.relu1 = nn.Identity()

            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   padding=(kernel_size - 1) * dilation, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_channels)
            #self.relu2 = nn.ReLU()
            #self.relu2 = nn.PReLU(out_channels)
            self.relu2 = nn.Identity()

            self.dilation = dilation
            self.kernel_size = kernel_size

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            return out[:, :, :-self.dilation * (self.kernel_size - 1)]  # causal trim

    class Transformer_encoder(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, dropout):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
            )

        def forward(self, x):
            x = x.permute(2, 0, 1)      # (B, C, L) -> (L, B, C)
            out = self.layer(x)
            return out.permute(1, 2, 0) # (L, B, C) -> (B, C, L)

    class TCN_Transformer_model(nn.Module):
        def __init__(self, input_dim, tcn_channels, kernel_size,
                     transformer_d_model, transformer_nhead, transformer_dim_feedforward,
                     transformer_dropout, output_dim):
            super().__init__()
            self.tcn_layers = nn.ModuleList()
            in_channels = input_dim
            for out_channels in tcn_channels:
                self.tcn_layers.append(TCN_block(in_channels, out_channels, kernel_size, dilation=2 ** len(self.tcn_layers)))
                in_channels = out_channels

            self.transformer = Transformer_encoder(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout
            )
            self.linear = nn.Linear(tcn_channels[-1], output_dim)

        def forward(self, x):
            # x: (B, C, L)
            for layer in self.tcn_layers:
                x = layer(x)
            x = self.transformer(x)
            out = self.linear(x[:, :, -1])  # last time step
            return out

    # --- model params (keep your second version) ---
    input_dim = config.INPUT_DIM
    tcn_channels =   [128, 256, 512]
    kernel_size = 2
    transformer_d_model = 512
    transformer_nhead = 16
    transformer_dim_feedforward = 128
    transformer_dropout = 0.2
    output_dim = output_timesteps

    model = TCN_Transformer_model(
        input_dim, tcn_channels, kernel_size,
        transformer_d_model, transformer_nhead,
        transformer_dim_feedforward, transformer_dropout,
        output_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=50, verbose=True
    )

    num_epochs = 1000
    train_losses, valid_losses = [], []
    best_valid_rmse = float('inf')

    best_model_path = f'./{output_dir}/{RatioInterval}_diff_best_model.pth'
    print(f"Best model path: {best_model_path}")

    # quick shape check
    for Xb, yb in train_loader:
        # Xb: (B, L, C)
        assert Xb.ndim == 3 and Xb.shape[2] == config.INPUT_DIM, \
            f"Train batch X shape mismatch: {Xb.shape}, expected (*, *, {config.INPUT_DIM})"
        assert yb.ndim == 3 and yb.shape[1] == output_timesteps and yb.shape[2] == 1, \
            f"Train batch y shape mismatch: {yb.shape}, expected (*, {output_timesteps}, 1)"
        break

    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0

        for X_batch, y_batch in train_loader:
            # X: (B, L, C) -> (B, C, L)
            X_batch = X_batch.permute(0, 2, 1).to(device)
            y_batch = y_batch.squeeze(-1).to(device)  # (B, N)

            optimizer.zero_grad()
            output = model(X_batch)                   # (B, N)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()
            train_total += torch.sqrt(loss).item()

        train_rmse = train_total / max(1, len(train_loader))
        train_losses.append(train_rmse)

        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.permute(0, 2, 1).to(device)
                y_batch = y_batch.squeeze(-1).to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                valid_total += torch.sqrt(loss).item()

        valid_rmse = valid_total / max(1, len(valid_loader))
        valid_losses.append(valid_rmse)

        scheduler.step(valid_rmse)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train RMSE: {train_rmse:.4f}  Valid RMSE: {valid_rmse:.4f}  LR: {optimizer.param_groups[0]['lr']:.8f}")

        # save best
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            torch.save(model.state_dict(), best_model_path)
            print("Valid RMSE improved, saving best model.")

        # always update loss csv
        pd.DataFrame({'Train_RMSE': train_losses, 'Valid_RMSE': valid_losses}).to_csv(
            f'./{output_dir}/{RatioInterval}_diff_best_model_loss.csv', index=False
        )

    print(f"Best Valid RMSE: {best_valid_rmse:.4f}")
    print(f"Best model saved to: {best_model_path}")


def Ratio_data_split(site_sets, output_dir, ratio=0.8, avoid_interval=1, input_step=72):
    all_train_subsets = []
    all_valid_subsets = []

    # 2016 leap-year month days
    days_per_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [d * 24 for d in days_per_month]

    month_hour_start = [0]
    for h in hours_per_month:
        month_hour_start.append(month_hour_start[-1] + h)

    for _s in site_sets:
        dataset = torch.load(f"./{output_dir}/{_s}_diff_training_data.pt", weights_only=False)
        N = len(dataset)

        if avoid_interval != 0:
            m = avoid_interval  # 1~12
            H_start = month_hour_start[m - 1]
            H_end = month_hour_start[m]

            # t0 = input_step + d  (we keep t0=i in datasetdiff)

            ms = month_hour_start[m - 1]
            me = month_hour_start[m]

            # --------- (1) train/process 排除區：t0 ∈ [ms, me+S) -> d ∈ [ms-S, me)
            d_excl_start = max(0, ms - input_step)
            d_excl_end   = min(N, me)
            exclude_for_train = set(range(d_excl_start, d_excl_end))
            
            # --------- (2) avoid_predict cold-start：t0 ∈ [ms+S, me) -> d ∈ [ms, me-S)
            d_avoid_start = max(0, ms)
            d_avoid_end   = min(N, me - input_step)
            avoid_for_predict = set(range(d_avoid_start, d_avoid_end))

            all_indices_set = set(range(N))

            # train/process 用：拿掉 exclude_for_train
            process_indices = sorted(list(all_indices_set - exclude_for_train))

            # avoid_predict 用：只保留 cold-start 那段
            avoid_save_indices = sorted(list(avoid_for_predict))

            avoid_dataset = Subset(dataset, process_indices)
            ratio_avoid_dataset = Subset(dataset, avoid_save_indices)

            torch.save(ratio_avoid_dataset, f"./{output_dir}/{_s}_ratio_avoid.pt")
            print(f"Save ratio avoid data: ./{output_dir}/{_s}_ratio_avoid.pt")
        else:
            avoid_dataset = dataset

        train_size = int(ratio * len(avoid_dataset))
        valid_size = len(avoid_dataset) - train_size
        current_train_subset, current_valid_subset = random_split(avoid_dataset, [train_size, valid_size])

        
        torch.save(avoid_dataset, f"./{output_dir}/{_s}_ratio_process.pt")
        print(f"Save ratio process data: ./{output_dir}/{_s}_ratio_process.pt")

        all_train_subsets.append(current_train_subset)
        all_valid_subsets.append(current_valid_subset)

    combined_train_dataset = ConcatDataset(all_train_subsets) if all_train_subsets else []
    combined_valid_dataset = ConcatDataset(all_valid_subsets) if all_valid_subsets else []

    batch_size = 16
    # ✅ 你確認要 shuffle=True
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True) if combined_train_dataset else DataLoader([])
    valid_loader = DataLoader(combined_valid_dataset, batch_size=batch_size, shuffle=False) if combined_valid_dataset else DataLoader([])

    print(f"Train_sets : {len(combined_train_dataset)}")
    print(f"Valid_sets : {len(combined_valid_dataset)}")
    return train_loader, valid_loader


def Interval_data_split(site_sets, time_steps, output_dir, train_intervals, avoid_interval=0):
    all_train_subsets = []
    all_valid_subsets = []

    days_per_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [d * 24 for d in days_per_month]
    month_hour_start = [0]
    for h in hours_per_month:
        month_hour_start.append(month_hour_start[-1] + h)

    for _s in site_sets:
        dataset = torch.load(f"./{output_dir}/{_s}_diff_training_data.pt", weights_only=False)
        dataset_size = len(dataset)

        if avoid_interval != 0:
            m = avoid_interval
            H_start = month_hour_start[m - 1]
            H_end = month_hour_start[m]
            d_start_raw = H_start - time_steps
            d_end_raw = H_end - time_steps
            d_start = max(0, d_start_raw)
            d_end = min(dataset_size, d_end_raw)
            avoid_indices_set = set(range(d_start, d_end))
        else:
            avoid_indices_set = set()

        train_indices = []
        for start, end in train_intervals:
            actual_start = max(0, start)
            actual_end = min(dataset_size, end)
            if actual_start < actual_end:
                for idx in range(actual_start, actual_end):
                    if idx not in avoid_indices_set:
                        train_indices.append(idx)

        train_indices = sorted(list(set(train_indices)))
        all_indices_set = set(range(dataset_size))
        valid_indices_set = all_indices_set - set(train_indices) - avoid_indices_set

        valid_indices = sorted(list(valid_indices_set))
        avoid_indices = sorted(list(avoid_indices_set))

        current_train_subset = Subset(dataset, train_indices)
        current_valid_subset = Subset(dataset, valid_indices)
        current_avoid_subset = Subset(dataset, avoid_indices) if avoid_indices else None

        torch.save(current_valid_subset, f"./{output_dir}/{_s}_interval_valid.pt")
        print(f"Save interval valid data: ./{output_dir}/{_s}_interval_valid.pt")

        if current_avoid_subset is not None and len(current_avoid_subset) > 0:
            torch.save(current_avoid_subset, f"./{output_dir}/{_s}_interval_avoid.pt")
            print(f"Save interval avoid data: ./{output_dir}/{_s}_interval_avoid.pt")

        if len(current_train_subset) > 0:
            all_train_subsets.append(current_train_subset)
        if len(current_valid_subset) > 0:
            all_valid_subsets.append(current_valid_subset)

    combined_train_dataset = ConcatDataset(all_train_subsets) if all_train_subsets else []
    combined_valid_dataset = ConcatDataset(all_valid_subsets) if all_valid_subsets else []

    batch_size = 16
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True) if combined_train_dataset else DataLoader([])
    valid_loader = DataLoader(combined_valid_dataset, batch_size=batch_size, shuffle=False) if combined_valid_dataset else DataLoader([])
    print(f"合併後總訓練集大小: {len(combined_train_dataset)}")
    print(f"合併後總驗證集大小: {len(combined_valid_dataset)}")
    return train_loader, valid_loader


if __name__ == "__main__":
    args = arg()
    site_list = json.loads(args.site_sets)

    # ----- ratio -----
    train_loader, valid_loader = Ratio_data_split(
        site_sets=site_list,
        ratio=0.8,
        output_dir=args.output_dir,
        avoid_interval=args.Avoid_interval,
        input_step=args.Input_timestep
    )
    train(
        site_sets=site_list,
        time_steps=args.Input_timestep,
        output_timesteps=args.Output_timestep,
        output_dir=args.output_dir,
        train_loader=train_loader,
        valid_loader=valid_loader,
        RatioInterval="ratio"
    )

    # ----- interval -----
    mounth_days = [30, 31, 29]
    train_days = 20
    month_hour_data = [
        (0, 720 - args.Input_timestep - (mounth_days[1] - train_days) * 24),
        (720 - args.Input_timestep, 1416 - args.Input_timestep - (mounth_days[2] - train_days) * 24),
        (1416 - args.Input_timestep, 2160 - args.Input_timestep - (mounth_days[1] - train_days) * 24),
        (2160 - args.Input_timestep, 2880 - args.Input_timestep - (mounth_days[0] - train_days) * 24),
        (2880 - args.Input_timestep, 3624 - args.Input_timestep - (mounth_days[1] - train_days) * 24),
        (3624 - args.Input_timestep, 4344 - args.Input_timestep - (mounth_days[0] - train_days) * 24),
        (4344 - args.Input_timestep, 5088 - args.Input_timestep - (mounth_days[1] - train_days) * 24),
        (5088 - args.Input_timestep, 5832 - args.Input_timestep - (mounth_days[1] - train_days) * 24),
        (5832 - args.Input_timestep, 6552 - args.Input_timestep - (mounth_days[0] - train_days) * 24),
        (6552 - args.Input_timestep, 7296 - args.Input_timestep - (mounth_days[1] - train_days) * 24),
        (7296 - args.Input_timestep, 8016 - args.Input_timestep - (mounth_days[0] - train_days) * 24),
        (8016 - args.Input_timestep, 8784 - args.Input_timestep - args.Output_timestep - (mounth_days[1] - train_days) * 24),
    ]

    train_loader, valid_loader = Interval_data_split(
        site_sets=site_list,
        time_steps=args.Input_timestep,
        output_dir=args.output_dir,
        train_intervals=month_hour_data,
        avoid_interval=args.Avoid_interval
    )
    train(
        site_sets=site_list,
        time_steps=args.Input_timestep,
        output_timesteps=args.Output_timestep,
        output_dir=args.output_dir,
        train_loader=train_loader,
        valid_loader=valid_loader,
        RatioInterval="interval"
    )
