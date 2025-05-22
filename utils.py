# ✅ utils.py (최신 리팩토링 버전)
# custom 데이터셋 기반 함수들만 유지하며 demo_mamba.py와 연동

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from skimage.transform import resize
from sklearn.metrics import confusion_matrix

# 시드 고정
def setup_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# ENVI 이미지 로딩
def load_envi_image(path_prefix):
    hdr_path = path_prefix.replace(".raw", ".hdr")
    with open(hdr_path, "r") as f:
        lines = f.readlines()
    meta = {k.strip(): v.strip().strip('{}') for line in lines if '=' in line for k, v in [line.split('=')]}
    samples = int(meta["samples"])
    lines = int(meta["lines"])
    bands = int(meta["bands"])
    dtype = np.uint16 if int(meta["data type"]) == 12 else np.float32
    interleave = meta.get("interleave", "bil")
    data = np.fromfile(path_prefix, dtype=dtype)
    if interleave == "bil":
        data = data.reshape((lines, bands, samples)).transpose((0, 2, 1))
    elif interleave == "bsq":
        data = data.reshape((bands, lines, samples)).transpose((1, 2, 0))
    elif interleave == "bip":
        data = data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unsupported interleave format: {interleave}")
    return data.astype(np.float32)

# White/Dark Reference 정규화
def apply_white_dark_reference(raw, white, dark):
    if raw.shape != white.shape:
        white = resize(white, raw.shape, preserve_range=True).astype(np.float32)
        dark = resize(dark, raw.shape, preserve_range=True).astype(np.float32)
    corrected = (raw - dark) / (white - dark + 1e-8)
    return np.clip(corrected, 0, 1)

# HSI 로딩 메인 함수
def load_HSI(args):
    base = args.data_dir
    sample = args.sample_name
    raw_path = os.path.join(base, "capture", f"{sample}.raw")
    white_path = os.path.join(base, "capture", f"WHITEREF_{sample}.raw")
    dark_path = os.path.join(base, "capture", f"DARKREF_{sample}.raw")
    label_path = os.path.join(base, "capture", "label.npy")

    raw = load_envi_image(raw_path)
    white = load_envi_image(white_path)
    dark = load_envi_image(dark_path)
    input = apply_white_dark_reference(raw, white, dark)
    label2d = np.load(label_path).astype(np.int32)

    # 🔁 명시적 라벨 remap
    label_remap = np.full_like(label2d, fill_value=-1)
    label_remap[label2d == 1] = 0  # 정상
    label_remap[label2d == 2] = 1  # DW
    label_remap[label2d == 3] = 2  # DA
    label_remap[label2d == 4] = 3  # 기타

    rng = np.random.default_rng(args.seed)
    valid = np.argwhere(label_remap >= 0)
    rng.shuffle(valid)
    split = int(len(valid) * 0.7)

    # Train/Test 라벨 생성
    TR = np.full_like(label_remap, fill_value=-1)
    TE = np.full_like(label_remap, fill_value=-1)
    TR[tuple(valid[:split].T)] = label_remap[tuple(valid[:split].T)]
    TE[tuple(valid[split:].T)] = label_remap[tuple(valid[split:].T)]
    label = np.where(TR >= 0, TR, 0) + np.where(TE >= 0, TE, 0)  # 병합 시 -1 방지

    num_classes = 4  # 고정

    palette = [
        [0, 0, 0], [83,171,72], [137,186,67], [66,132,91], [60,131,69],
        [144,82,54], [105,188,200], [255,255,255], [199,176,201], [218,51,44],
        [119,35,36], [55,101,166], [224,219,84], [217,142,52], [84,48,126],
        [227,119,91], [157,87,150]
    ]
    color_matrix = [c for c in palette[:num_classes+1]]
    color_matrix_pred = color_matrix[1:]
    color_matrix = [[v/256. for v in rgb] for rgb in color_matrix]
    color_matrix_pred = [[v/256. for v in rgb] for rgb in color_matrix_pred]

    return input, label, num_classes, TR, TE, color_matrix, color_matrix_pred


# 학습/테스트 인덱스 생성

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train, pos_train = [], {}
    number_test, pos_test = [], {}
    number_true, pos_true = [], {}

    for i in range(num_classes):
        pos_train[i] = np.argwhere(train_data == i)      # ✅ 올바른 비교
        number_train.append(len(pos_train[i]))
    total_pos_train = np.vstack([pos_train[i] for i in range(num_classes)])

    for i in range(num_classes):
        pos_test[i] = np.argwhere(test_data == i)        # ✅ 올바른 비교
        number_test.append(len(pos_test[i]))
    total_pos_test = np.vstack([pos_test[i] for i in range(num_classes)])

    for i in range(num_classes+1):                       # 배경까지 포함
        pos_true[i] = np.argwhere(true_data == i)
        number_true.append(len(pos_true[i]))
    total_pos_true = np.vstack([pos_true[i] for i in range(num_classes+1)])

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

# mirror padding

def mirror_hsi(height, width, band, input_normalize, patch=5):
    pad = patch // 2
    out = np.pad(input_normalize, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    print("**************************************************")
    print(f"patch is : {patch}")
    print(f"mirror_image shape : {out.shape}")
    print("**************************************************")
    return out

# patch 추출

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x, y = point[i]
    return mirror_image[x:x+patch, y:y+patch, :]

from collections import defaultdict

def balanced_sample_points(points, label_map, max_per_class=1000):
    class_points = defaultdict(list)
    for pt in points:
        x, y = pt
        cls = label_map[x, y]
        if cls >= 0:
            class_points[cls].append(pt)

    sampled = []
    for cls, pts in class_points.items():
        np.random.shuffle(pts)
        sampled.extend(pts[:max_per_class])
    return np.array(sampled)

def train_and_test_data(mirror_image, band, train_point, test_point, true_point,
                        patch=5, train_label_map=None, max_per_class=1000):
    
    # ▶ train_point 샘플링 적용
    if train_label_map is not None:
        sampled_train_point = balanced_sample_points(train_point, train_label_map, max_per_class)
    else:
        sampled_train_point = train_point  # 샘플링 없이 그대로 사용

    # ▶ 패치 추출
    x_train = np.array([
        gain_neighborhood_pixel(mirror_image, sampled_train_point, i, patch)
        for i in range(len(sampled_train_point))
    ])
    x_test = np.array([
        gain_neighborhood_pixel(mirror_image, test_point, i, patch)
        for i in range(len(test_point))
    ])

    print(f"x_train shape = {x_train.shape}, x_test shape = {x_test.shape}")
    return x_train, x_test, sampled_train_point

# 라벨

def train_and_test_label(train_point, test_point, true_point, train_label_map, test_label_map, full_label_map):
    def extract_labels(points, label_map):
        return np.array([label_map[x, y] for x, y in points])

    y_train = extract_labels(train_point, train_label_map)
    y_test  = extract_labels(test_point, test_label_map)
    y_true  = extract_labels(true_point, full_label_map)

    print(f"y_train: shape = {y_train.shape}, y_test: shape = {y_test.shape}, y_true: shape = {y_true.shape}")
    return y_train, y_test, y_true


# 평가 지표

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = matrix.shape
    correct = np.diag(matrix).sum()
    total = matrix.sum()
    OA = correct / total
    AA = np.diag(matrix) / matrix.sum(axis=1)
    AA_mean = np.mean(AA)
    pe = np.sum(matrix.sum(0) * matrix.sum(1)) / (total ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

# 학습 루프

class AvgrageMeter:
    def __init__(self): self.reset()
    def reset(self): self.avg = self.sum = self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum().mul_(100.0 / target.size(0)) for k in topk], target, pred.squeeze()

def train_epoch(model, loader, criterion, optimizer):
    objs, top1 = AvgrageMeter(), AvgrageMeter()
    tar, pre = [], []

    printed = False  # ✅ 분포 출력은 1회만
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        
        if not printed:
            from collections import Counter
            print("[DEBUG] y 클래스 분포 (train):", Counter(y.cpu().numpy()))
            printed = True
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        prec1, t, p = accuracy(out, y)
        objs.update(loss.item(), x.size(0))
        top1.update(prec1[0].item(), x.size(0))
        tar.extend(t.cpu().numpy())
        pre.extend(p.cpu().numpy())
    
    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, loader, criterion, optimizer):
    model.eval()
    top1 = AvgrageMeter()
    tar, pre = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            prec1, t, p = accuracy(out, y)
            top1.update(prec1[0].item(), x.size(0))
            tar.extend(t.cpu().numpy())
            pre.extend(p.cpu().numpy())
    return tar, pre
