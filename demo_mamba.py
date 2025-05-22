import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch import optim
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from s2mamba import S2Mamba
from utils import (
    setup_seed, load_HSI, chooose_train_and_test_point,
    mirror_hsi, train_and_test_data, train_and_test_label,
    train_epoch, valid_epoch, output_metric
)

# ───────────────────────
# Argument 설정
# ───────────────────────
parser = argparse.ArgumentParser("HSI Trainer (Multi-Sample)")
parser.add_argument('--dataset', choices=['custom'], default='custom')
parser.add_argument('--flag', choices=['test', 'train'], default='train')
parser.add_argument('--sess', default='s2mamba')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_freq', type=int, default=5)
parser.add_argument('--patches', type=int, default=7)
parser.add_argument('--epoches', type=int, default=400)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--data-dir', required=True, help='여러 샘플 폴더가 포함된 상위 폴더')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--pin-memory', action='store_true')
parser.add_argument('--drop-last', action='store_true')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
setup_seed(args)

# ───────────────────────
# 학습/테스트 대상 전체 샘플 로딩
# ───────────────────────
sample_folders = sorted(glob.glob(os.path.join(args.data_dir, "FX10e_TEST_*/")))
all_inputs, all_labels, all_TR, all_TE = [], [], [], []

for folder in sample_folders:
    sample_name = os.path.basename(os.path.normpath(folder))
    args.sample_name = sample_name
    input_normalize, label, num_classes, TR, TE, color_matrix, color_matrix_pred = load_HSI(args)
    all_inputs.append(input_normalize)
    all_labels.append(label)
    all_TR.append(TR)
    all_TE.append(TE)

input_normalize = np.concatenate(all_inputs, axis=0)
label = np.concatenate(all_labels, axis=0)
TR = np.concatenate(all_TR, axis=0)
TE = np.concatenate(all_TE, axis=0)
H, W, C = input_normalize.shape

print(f"height={H}, width={W}, band={C}")
print(f"[DEBUG] 전체 클래스 수 (remap 기준): {num_classes}")
print(f"[DEBUG] 라벨 내 존재하는 클래스 (remap): {np.unique(label[label >= 0])}")

# ───────────────────────
# 라벨 분리 및 patch 추출
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

# ───────────────────────
# Tensor 변환 및 DataLoader 구성
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
# 모델 및 최적화 설정
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

counter = Counter(y_train.cpu().numpy())
class_counts = np.array([counter.get(i, 1) for i in range(num_classes)])
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum()
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()

criterion = nn.CrossEntropyLoss(weight=weight_tensor).cuda()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

# ───────────────────────
# 학습 또는 테스트 수행
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
