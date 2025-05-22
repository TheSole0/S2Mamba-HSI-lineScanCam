# S²Mamba
논문 **"S²Mamba: A Spatial-spectral State Space Model for Hyperspectral Image Classification"**의 구현입니다.

본 저장소는 실제 산업 현장의 **금속 소재/피복관 결함 검출**을 위한 라벨링된 픽셀기반 **초분광 이상탐지 모델**로 확장 적용할 수 있도록 구성되었습니다.

---

## 📦 가상환경 및 설치

```bash
# 1. Conda 가상환경 생성
conda create -n s2mamba python=3.9 -y
conda activate s2mamba

# 2. PyTorch 및 필수 라이브러리 설치 (CUDA 11.8 기준)
pip install torch==2.6.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn einops matplotlib tqdm h5py
```

---

## 데이터 구조 예시

각 샘플 폴더는 아래와 같은 파일 구조를 따라야 합니다:

```
<루트 경로>/FX10e_TEST_YYYY-MM-DD_HH-MM-SS/
├── FX10e_TEST_*.hdr           # 주 영상 메타정보 (ENVI)
├── FX10e_TEST_*.raw           # 주 영상 데이터 (H x W x C)
├── DARKREF_*.hdr/.raw         # 다크 기준 영상
├── WHITEREF_*.hdr/.raw        # 화이트 기준 영상
├── label.npy                  # 정답 라벨맵 (H x W), npy 포맷
```

---
## 시각화 예시

<p align="center">
  <img src="figures/FX10e_TEST_2025-05-14_07-11-11_preview.png" alt="샘플1" width="45%"/>
  <img src="figures/FX10e_TEST_2025-05-14_07-12-17_preview.png" alt="샘플2" width="45%"/>
</p>

---

## 훈련 실행 예시

```bash
CUDA_VISIBLE_DEVICES=1 python demo_mamba.py \
  --data-dir "/workspace/NAS/home/crew/jinho/vision/Base_dataset/250514-level4(피복관)-2" \
  --flag train \
  --epoches 10 \
  --batch_size 4096
```

> `--data-dir`은 `.raw`, `.hdr`, `label.npy`가 포함된 상위 디렉토리 경로입니다.  
> 학습 로그 및 모델은 `outputs/` 하위 디렉토리에 저장됩니다.

---

## 🔍 추론 실행 예시

```bash
CUDA_VISIBLE_DEVICES=0 python predict_mamba.py \
  --data-dir "/workspace/NAS/home/crew/jinho/vision/Base_dataset/250514-level4(피복관)-2" \
  --sample-name FX10e_TEST_2025-05-14_07-11-11 \
  --batch-size 2048
```

> 추론 결과는 `./outputs/<sample-name>/` 경로에 저장되며, 클래스 맵과 확률 맵을 포함할 수 있습니다.

---

## 인용 정보

이 모델이 도움이 되셨다면, 아래 논문을 인용해주세요:

```bibtex
@ARTICLE{s2mamba,
  author={Wang, Guanchun and Zhang, Xiangrong and Peng, Zelin and Zhang, Tianyang and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={S2Mamba: A Spatial-spectral State Space Model for Hyperspectral Image Classification}, 
  year={2025},
  pages={1-1},
  doi={10.1109/TGRS.2025.3530993}}
```

---

## 참고 및 감사

본 프로젝트는 아래 오픈소스 프로젝트에 기반하고 있으며, 기여자 분들께 감사드립니다:

- [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [S2Mamba](https://github.com/PURE-melo/S2Mamba)
