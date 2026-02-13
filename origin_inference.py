# inference.py
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import origin_config as config


def arg():
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument('--Input_timestep', type=int, default=config.Input_timesteps, help='Input_timestep.')
    parser.add_argument('--Output_timestep', type=int, default=config.Output_timesteps, help='Output_timestep.')
    parser.add_argument('--Avoid_interval', type=int, default=7, help='Avoid_interval.')
    parser.add_argument('--site_sets', type=str, default=str(config.site_sets), help='site_sets.')
    parser.add_argument('--output_dir', type=str, default='results', help='Output folder name.')
    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def build_model(output_timesteps):
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
            return out[:, :, :-self.dilation * (self.kernel_size - 1)]

    class Transformer_encoder(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, dropout):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
            )

        def forward(self, x):
            x = x.permute(2, 0, 1)
            out = self.layer(x)
            return out.permute(1, 2, 0)

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
            for layer in self.tcn_layers:
                x = layer(x)
            x = self.transformer(x)
            return self.linear(x[:, :, -1])

    input_dim = config.INPUT_DIM
    tcn_channels = [128, 256, 512]
    kernel_size = 2
    transformer_d_model = 512
    transformer_nhead = 16
    transformer_dim_feedforward = 128
    transformer_dropout = 0.2

    model = TCN_Transformer_model(
        input_dim, tcn_channels, kernel_size,
        transformer_d_model, transformer_nhead,
        transformer_dim_feedforward, transformer_dropout,
        output_timesteps
    )
    return model


def predict(site, output_timesteps, output_dir, RatioInterval='ratio', Year_or_valid='diff_training_data'):
    model = build_model(output_timesteps).to(device)

    weight_path = f'./{output_dir}/{RatioInterval}_diff_best_model.pth'
    try:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[{site}] Error loading model weights: {weight_path}  ({e})")
        return

    model.eval()

    data_path = f"./{output_dir}/{site}_{Year_or_valid}.pt"
    try:
        predict_data = torch.load(data_path, weights_only=False)
    except Exception as e:
        print(f"[{site}] Error loading dataset: {data_path}  ({e})")
        return

    loader = DataLoader(predict_data, batch_size=16, shuffle=False)

    pred_chunks = []
    tru_chunks = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            # X: (B, L, C) -> (B, C, L)
            X_batch = X_batch.permute(0, 2, 1).to(device)
            out = model(X_batch)  # (B, N)
            pred_chunks.append(out.cpu().numpy())

            # y: (B, N, 1) -> (B, N)
            y_np = y_batch.squeeze(-1).cpu().numpy()
            tru_chunks.append(y_np)

    predictions = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros((0, output_timesteps))
    truth = np.concatenate(tru_chunks, axis=0) if tru_chunks else np.zeros((0, output_timesteps))

    # shape checks
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)

    if predictions.shape[1] != output_timesteps:
        print(f"[{site}] Prediction dim mismatch: {predictions.shape}, expected (*, {output_timesteps})")
        return
    if truth.shape[1] != output_timesteps:
        print(f"[{site}] Truth dim mismatch: {truth.shape}, expected (*, {output_timesteps})")
        return

    # length align
    min_len = min(predictions.shape[0], truth.shape[0])
    predictions = predictions[:min_len]
    truth = truth[:min_len]

    pred_cols = [f'Prediction_t+{i+1}' for i in range(output_timesteps)]
    truth_cols = [f'Truth_t+{i+1}' for i in range(output_timesteps)]

    combined = np.hstack((predictions, truth))
    pre_tru = pd.DataFrame(combined, columns=pred_cols + truth_cols)

    if Year_or_valid == 'diff_training_data':
        AorV = 'all'
    elif Year_or_valid in ('ratio_avoid', 'interval_avoid'):
        AorV = 'avoid'
    elif Year_or_valid in ('ratio_process', 'interval_process'):
        AorV = 'process'
    else:
        AorV = 'valid'

    save_path = f'./{output_dir}/{site}_{RatioInterval}_{AorV}_predict.csv'
    pre_tru.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    args = arg()
    site_list = json.loads(args.site_sets)

    for _s in site_list:
        # ratio
        predict(site=_s, output_timesteps=args.Output_timestep, output_dir=args.output_dir, RatioInterval='ratio', Year_or_valid='diff_training_data')
        predict(site=_s, output_timesteps=args.Output_timestep, output_dir=args.output_dir,
        RatioInterval='ratio', Year_or_valid='ratio_process')
        if args.Avoid_interval != 0:
            predict(site=_s, output_timesteps=args.Output_timestep, output_dir=args.output_dir, RatioInterval='ratio', Year_or_valid='ratio_avoid')

        # interval
        predict(site=_s, output_timesteps=args.Output_timestep, output_dir=args.output_dir, RatioInterval='interval', Year_or_valid='diff_training_data')
        predict(site=_s, output_timesteps=args.Output_timestep, output_dir=args.output_dir, RatioInterval='interval', Year_or_valid='interval_valid')
        if args.Avoid_interval != 0:
            predict(site=_s, output_timesteps=args.Output_timestep, output_dir=args.output_dir, RatioInterval='interval', Year_or_valid='interval_avoid')
