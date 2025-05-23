# predict_s2mamba.py — 상대경로 및 Jupyter 사용 가능 모듈화 버전
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .s2mamba import S2Mamba
from .s2mamba_utils import (
    load_HSI, setup_seed, mirror_hsi,
    train_and_test_data
)

def convert_to_rgb(hsi_img):
    rgb = hsi_img[:, :, [111, 82, 26]]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)

def overlay_prediction_on_rgb(rgb_img, pred_map, color_map, alpha=0.6):
    overlay = rgb_img.copy()
    for cls in np.unique(pred_map):
        if cls == -1:
            continue
        mask = pred_map == cls
        color = (np.array(mcolors.to_rgb(color_map(cls))) * 255).astype(np.uint8)
        for c in range(3):
            overlay[:, :, c][mask] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][mask]
            ).astype(np.uint8)
    return overlay

def run_predict_s2mamba(
    data_dir: str,
    sample_name: str,
    sess: str = "s2mamba",
    patches: int = 7,
    batch_size: int = 2048,
    seed: int = 0,
    ckpt: str = "best",
    device: str = "cuda:0"
):
    class Args: pass
    args = Args()
    args.data_dir = data_dir
    args.sample_name = sample_name
    args.seed = seed
    setup_seed(args)

    input, label, num_classes, TR, TE, color_matrix, _ = load_HSI(args)
    H, W, C = input.shape

    test_idx = np.argwhere(np.ones((H, W), dtype=bool))
    mirror = mirror_hsi(H, W, C, input, patch=patches)

    _, x_test, _ = train_and_test_data(
        mirror_image=mirror,
        band=C,
        train_point=test_idx,
        test_point=test_idx,
        true_point=test_idx,
        patch=patches,
        train_label_map=None,
        max_per_class=999999
    )

    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
    loader = Data.DataLoader(Data.TensorDataset(x_test), batch_size=batch_size, shuffle=False)

    model = S2Mamba(
        in_chans=C,
        patch=patches,
        num_classes=num_classes,
        depths=[1],
        dims=[64],
        drop_path_rate=0.4,
        attn_drop_rate=0.4
    ).to(device)

    ckpt_path = f"{sess}_custom_{ckpt}.pt"
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch_x, in loader:
            out = model(batch_x.to(device))
            preds = out.argmax(1).cpu().numpy()
            all_preds.append(preds)

    pred_all = np.concatenate(all_preds)
    pred_map = np.full((H, W), fill_value=-1, dtype=np.int32)
    for (r, c), pred in zip(test_idx, pred_all):
        pred_map[r, c] = pred

    rgb_img = convert_to_rgb(input)
    overlay_img = overlay_prediction_on_rgb(rgb_img, pred_map, plt.get_cmap("tab20"))

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1); plt.title("Input RGB"); plt.imshow(rgb_img); plt.axis("off")
    plt.subplot(2, 2, 2); plt.title("Ground Truth"); plt.imshow(label, cmap="tab20", vmin=0, vmax=num_classes - 1); plt.axis("off")
    plt.subplot(2, 2, 3); plt.title("Prediction (Full Area)"); plt.imshow(pred_map, cmap="tab20", vmin=0, vmax=num_classes - 1); plt.axis("off")
    plt.subplot(2, 2, 4); plt.title("Prediction + RGB Overlay"); plt.imshow(overlay_img); plt.axis("off")
    plt.tight_layout()

    save_path = os.path.join(data_dir, f"{sample_name}_predict_overlay.png")
    plt.savefig(save_path)
    print(f"[✅] 전체 시각화 저장 완료: {save_path}")
    return pred_map, save_path
