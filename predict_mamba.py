# predict_mamba.py
# ─────────────────────────────────────────────────
# S2Mamba 모델 기반 전체 영역 예측 + RGB 오버레이 시각화 저장

import os
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from s2mamba import S2Mamba
from utils import (
    load_HSI, setup_seed, mirror_hsi,
    train_and_test_data
)

# ───────────────────────────────
# RGB 및 오버레이 함수 정의
# ───────────────────────────────
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

# ───────────────────────────────
# Argument 설정
# ───────────────────────────────
parser = argparse.ArgumentParser("HSI Predictor")
parser.add_argument('--data-dir', required=True)
parser.add_argument('--sample-name', required=True)
parser.add_argument('--sess', default='s2mamba')
parser.add_argument('--patches', type=int, default=7)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ckpt', choices=['best', 'last'], default='best', help='불러올 모델 체크포인트')

args = parser.parse_args()

# ───────────────────────────────
# 1. 데이터 로딩 및 정규화
# ───────────────────────────────
setup_seed(args)
input, label, num_classes, TR, TE, color_matrix, _ = load_HSI(args)
H, W, C = input.shape

# ───────────────────────────────
# 2. 전체 영역 인덱스 기반 패치 추출
# ───────────────────────────────
test_idx = np.argwhere(np.ones((H, W), dtype=bool))  # 전체 픽셀 대상
mirror = mirror_hsi(H, W, C, input, patch=args.patches)

_, x_test, _ = train_and_test_data(
    mirror_image=mirror,
    band=C,
    train_point=test_idx,
    test_point=test_idx,
    true_point=test_idx,
    patch=args.patches,
    train_label_map=None,
    max_per_class=999999
)

# ───────────────────────────────
# 3. DataLoader 구성
# ───────────────────────────────
x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
loader = Data.DataLoader(Data.TensorDataset(x_test), batch_size=args.batch_size, shuffle=False)

# ───────────────────────────────
# 4. 모델 로드 및 추론
# ───────────────────────────────
model = S2Mamba(
    in_chans=C,
    patch=args.patches,
    num_classes=num_classes,
    depths=[1],
    dims=[64],
    drop_path_rate=0.4,
    attn_drop_rate=0.4
).cuda()

print(f"[INFO] 모델 출력 확인:")
model.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, C, args.patches, args.patches).cuda()
    print(f"→ output shape: {model(dummy_input).shape}")


ckpt_path = f'{args.sess}_custom_{args.ckpt}.pt'  # 예: s2mamba_custom_best.pt 또는 s2mamba_custom_last.pt
model.load_state_dict(torch.load(ckpt_path))


model.eval()

# ───────────────────────────────
# 5. 추론 수행
# ───────────────────────────────
all_preds = []
with torch.no_grad():
    for batch_x, in loader:
        out = model(batch_x.cuda())
        preds = out.argmax(1).cpu().numpy()
        all_preds.append(preds)

pred_all = np.concatenate(all_preds)

pred_map = np.full((H, W), fill_value=-1, dtype=np.int32)
for (r, c), pred in zip(test_idx, pred_all):
    pred_map[r, c] = pred

print(f"[DEBUG] 예측 클래스 종류: {np.unique(pred_all)}")

# ───────────────────────────────
# 6. 시각화 및 저장
# ───────────────────────────────
rgb_img = convert_to_rgb(input)
overlay_img = overlay_prediction_on_rgb(rgb_img, pred_map, plt.get_cmap("tab20"))

plt.figure(figsize=(12, 10))

# ① Input RGB
plt.subplot(2, 2, 1)
plt.title("Input RGB")
plt.imshow(rgb_img)
plt.axis("off")

# ② GT (Label)
plt.subplot(2, 2, 2)
plt.title("Ground Truth")
plt.imshow(label, cmap="tab20", vmin=0, vmax=num_classes - 1)
plt.axis("off")

# ③ Prediction
plt.subplot(2, 2, 3)
plt.title("Prediction (Full Area)")
plt.imshow(pred_map, cmap="tab20", vmin=0, vmax=num_classes - 1)
plt.axis("off")

# ④ RGB Overlay
plt.subplot(2, 2, 4)
plt.title("Prediction + RGB Overlay")
plt.imshow(overlay_img)
plt.axis("off")

plt.tight_layout()

# 저장
save_path = os.path.join(args.data_dir, f"{args.sample_name}_predict_overlay.png")
plt.savefig(save_path)
print(f"[✅] 전체 시각화 저장 완료: {save_path}")
