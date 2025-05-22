import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from s2mamba import S2Mamba
from utils import *
from utils import (
    setup_seed, load_HSI, chooose_train_and_test_point,
    mirror_hsi, train_and_test_data, train_and_test_label,
    train_epoch, valid_epoch, output_metric
)

# ───────────────────────
# Argument 설정
# ───────────────────────
parser = argparse.ArgumentParser("HSI Trainer (Custom Only)")
parser.add_argument('--dataset', choices=['custom'], default='custom')
parser.add_argument('--flag', choices=['test', 'train'], default='train')
parser.add_argument('--sess', default='s2mamba')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--patches', type=int, default=7)
parser.add_argument('--epoches', type=int, default=400)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--data-dir', required=True, help='샘플 폴더 경로 (capture 포함)')
parser.add_argument('--sample-name', required=True, help='raw/white/dark 공통 prefix (예: FX10e_2025...)')
parser.add_argument('--num-workers', type=int, default=4, help='DataLoader 병렬 처리 개수')
parser.add_argument('--pin-memory', action='store_true', help='DataLoader GPU 전송 최적화')
parser.add_argument('--drop-last', action='store_true', help='마지막 배치 제거 여부')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# ───────────────────────
# 1. Seed 고정 및 데이터 로딩
# ───────────────────────
setup_seed(args)
input_normalize, label, num_classes, TR, TE, color_matrix, color_matrix_pred = load_HSI(args)
H, W, C = input_normalize.shape
print(f"height={H}, width={W}, band={C}")

# ───────────────────────
# 🔍 클래스 분포 디버깅 (remap 기준)
# ───────────────────────
print(f"[DEBUG] 전체 클래스 수 (remap 기준): {num_classes}")
print(f"[DEBUG] 라벨 내 존재하는 클래스 (remap): {np.unique(label[label >= 0])}")

train_ids, train_counts = np.unique(TR[TR > 0], return_counts=True)
test_ids, test_counts = np.unique(TE[TE > 0], return_counts=True)

class_names_remap = {
    0: "정상",  # Remap 기준
    1: "DW",
    2: "DA",
    3: "기타"
}
print("라벨 고유값:", np.unique(label))
print(f"[DEBUG] ▶ Train 클래스 분포 (remap):")
for cid, cnt in zip(train_ids, train_counts):
    cname = class_names_remap.get(cid, f"Class {cid}")
    print(f"  Class {cid} ({cname}): {cnt}개")

print(f"[DEBUG] ▶ Test 클래스 분포 (remap):")
for cid, cnt in zip(test_ids, test_counts):
    cname = class_names_remap.get(cid, f"Class {cid}")
    print(f"  Class {cid} ({cname}): {cnt}개")


# ───────────────────────
# 2. 라벨 분리 및 patch 추출
# ───────────────────────
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror = mirror_hsi(H, W, C, input_normalize, patch=args.patches)
x_train, x_test, sampled_train_point = train_and_test_data(
    mirror, C,
    train_point=total_pos_train,
    test_point=total_pos_test,
    true_point=total_pos_true,
    patch=args.patches,
    train_label_map=TR,
    max_per_class=1000
)
y_train, y_test, y_true = train_and_test_label(
    train_point=sampled_train_point,
    test_point=total_pos_test,
    true_point=total_pos_true,
    train_label_map=TR,
    test_label_map=TE,
    full_label_map=label
)

# GT 시각화 저장
plt.imsave(os.path.join(args.data_dir, f"{args.sample_name}_GT.png"), label, cmap="tab20", vmin=0, vmax=num_classes-1)

# ───────────────────────
# 3. Tensor 변환 및 Dataloader 구성
# ───────────────────────
x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).float()
x_test  = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
y_train = torch.from_numpy(y_train).long()
y_test  = torch.from_numpy(y_test).long()

train_loader = Data.DataLoader(
    Data.TensorDataset(x_train, y_train),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    drop_last=args.drop_last
)

test_loader = Data.DataLoader(
    Data.TensorDataset(x_test, y_test),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    drop_last=args.drop_last
)


# ───────────────────────
# 4. 모델 설정
# ───────────────────────
model = S2Mamba(
    in_chans=C,
    patch=args.patches,
    num_classes=num_classes,
    depths=[1],
    dims=[64],
    drop_path_rate=args.dropout,
    attn_drop_rate=args.dropout
).cuda()

from collections import Counter

counter = Counter(y_train.cpu().numpy())
class_counts = np.array([counter.get(i, 1) for i in range(num_classes)])
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum()
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()

criterion = nn.CrossEntropyLoss(weight=weight_tensor).cuda()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

# ───────────────────────
# 5. 학습 or 테스트
# ───────────────────────
if args.flag == 'test':
    model = torch.load(f'./{args.sess}_{args.dataset}.pt').cuda()
    model.eval()
    tar_v, pre_v = valid_epoch(model, test_loader, criterion, optimizer)
    OA, AA_mean, Kappa, AA = output_metric(tar_v, pre_v)
    print(f"[TEST] OA={OA:.4f}, AA={AA_mean:.4f}, Kappa={Kappa:.4f}")

elif args.flag == 'train':
    OA_ls = []
    best_OA = 0.0
    best_path = f'{args.sess}_{args.dataset}_best.pt'

    for epoch in range(args.epoches):
        scheduler.step()
        model.train()
        train_acc, train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer)
        print(f"[Epoch {epoch:03d}] TrainAcc={train_acc:.4f} | Loss={train_loss:.4f}")

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, test_loader, criterion, optimizer)
            OA, AA_mean, Kappa, _ = output_metric(tar_v, pre_v)
            print(f"[VALID] OA={OA:.4f} | AA={AA_mean:.4f} | Kappa={Kappa:.4f}")
            OA_ls.append(OA.item())

            if OA > best_OA:
                best_OA = OA
                torch.save(model.state_dict(), best_path)
                print(f"[SAVE] Best model updated: OA={best_OA:.4f} -> saved to '{best_path}'")

    torch.save(model, f'{args.sess}_{args.dataset}.pt')
    print(f"[DONE] Final OA={OA:.4f} | AA={AA_mean:.4f} | Kappa={Kappa:.4f}")
