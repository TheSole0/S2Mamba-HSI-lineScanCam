# train_s2mamba.py — 상대경로 import 버전 (Jupyter/모듈화용)
import os
import glob
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch import optim

from .s2mamba import S2Mamba
from .s2mamba_utils import (
    setup_seed, load_HSI, chooose_train_and_test_point,
    mirror_hsi, train_and_test_data, train_and_test_label,
    train_epoch, valid_epoch, output_metric
)

def run_train_s2mamba(
    data_dir: str,
    sess: str = "s2mamba",
    seed: int = 0,
    batch_size: int = 64,
    test_freq: int = 5,
    patches: int = 7,
    epoches: int = 400,
    lr: float = 5e-4,
    gamma: float = 0.99,
    weight_decay: float = 5e-3,
    dropout: float = 0.4,
    num_workers: int = 4,
    pin_memory: bool = False,
    drop_last: bool = False,
    device: str = "cuda:0"
):
    torch.cuda.set_device(device)
    class Args: pass
    args = Args()
    args.seed = seed
    args.data_dir = data_dir  # 반드시 포함
    setup_seed(args)

    sample_folders = sorted(glob.glob(os.path.join(data_dir, "FX10e_TEST_*/")))
    all_inputs, all_labels, all_TR, all_TE = [], [], [], []

    for folder in sample_folders:
        args.sample_name = os.path.basename(os.path.normpath(folder))
        input_normalize, label, num_classes, TR, TE, _, _ = load_HSI(args)
        all_inputs.append(input_normalize)
        all_labels.append(label)
        all_TR.append(TR)
        all_TE.append(TE)

    input_normalize = np.concatenate(all_inputs, axis=0)
    label = np.concatenate(all_labels, axis=0)
    TR = np.concatenate(all_TR, axis=0)
    TE = np.concatenate(all_TE, axis=0)
    H, W, C = input_normalize.shape

    total_pos_train, total_pos_test, total_pos_true, _, _, _ = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror = mirror_hsi(H, W, C, input_normalize, patch=patches)
    x_train, x_test, sampled_train_point = train_and_test_data(mirror, C, total_pos_train, total_pos_test, total_pos_true, patch=patches, train_label_map=TR)
    y_train, y_test, y_true = train_and_test_label(sampled_train_point, total_pos_test, total_pos_true, TR, TE, label)

    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).float()
    x_test  = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
    y_train = torch.from_numpy(y_train).long()
    y_test  = torch.from_numpy(y_test).long()

    train_loader = Data.DataLoader(Data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    test_loader  = Data.DataLoader(Data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    model = S2Mamba(
        in_chans=C,
        patch=patches,
        num_classes=num_classes,
        depths=[1],
        dims=[64],
        drop_path_rate=dropout,
        attn_drop_rate=dropout
    ).to(device)

    class_counts = np.array([Counter(y_train.cpu().numpy()).get(i, 1) for i in range(num_classes)])
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    OA_ls = []
    best_OA = 0.0
    best_path = f"{sess}_custom_best.pt"

    for epoch in range(epoches):
        scheduler.step()
        model.train()
        train_acc, train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer)
        print(f"[Epoch {epoch:03d}] TrainAcc={train_acc:.4f} | Loss={train_loss:.4f}")

        if (epoch % test_freq == 0) or (epoch == epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, test_loader, criterion, optimizer)
            OA, AA_mean, Kappa, _ = output_metric(tar_v, pre_v)
            print(f"[VALID] OA={OA:.4f} | AA={AA_mean:.4f} | Kappa={Kappa:.4f}")
            OA_ls.append(OA.item())

            if OA > best_OA:
                best_OA = OA
                torch.save(model.state_dict(), best_path)
                print(f"[SAVE] Best model updated: OA={best_OA:.4f} -> saved to '{best_path}'")

    final_path = f"{sess}_custom.pt"
    torch.save(model, final_path)
    print(f"[DONE] Final OA={OA:.4f} | AA={AA_mean:.4f} | Kappa={Kappa:.4f}")
    return model, best_path, final_path
