# S2A-Net Training Guide for Few-Shot Object Detection

## 🎯 Overview
This guide documents the successful setup and training of S2A-Net for 12-class base training, preparing for Few-Shot Object Detection. The setup uses MMRotate (official implementation) with MMDetection 2.28.2 for compatibility.

## ✅ What We Accomplished

### 1. **Environment Setup** ✅
- **PyTorch 1.10 + cu113** (compatible with RTX 4070)
- **mmcv-full 1.6.2** (pre-built, no compilation needed)
- **mmdet 2.28.2** (compatible version)
- **mmrotate 0.3.4** (official S2A-Net implementation)

### 2. **S2A-Net Configuration** ✅
- **12 base classes** (excluding plane, baseball-diamond, tennis-court)
- **FAM head + ODM head** (exactly as per paper)
- **All hyperparameters** matching the paper:
  - SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)
  - 12 epochs with LR drops at 8 & 11
  - Batch size 2 (optimized for Windows)
  - 1024×1024 images with random flip/rotation

### 3. **Dataset Ready** ✅
- **7,006 training samples** in DOTA format
- **Proper structure**: `data/base_training/images/` + `data/base_training/labelTxt/`
- **12-class filtering** configured in the model

### 4. **Training Successfully Running** ✅
- **Stable training** with losses around 4.0-4.6
- **Healthy gradients** (norms 2-3)
- **Both FAM and ODM heads** training properly

## 🚀 Quick Start for Teammates

### Prerequisites
- Windows 10/11
- NVIDIA GPU (RTX 4070 tested)
- Anaconda/Miniconda

### Step 1: Environment Setup
```bash
# Create conda environment
conda create -n s2anet python=3.8
conda activate s2anet

# Install PyTorch (compatible with RTX 4070)
conda install pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0 cudatoolkit=11.3 -c pytorch

# Install MMCV and MMDetection
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.28.2

# Install MMRotate (contains S2A-Net)
pip install mmrotate==0.3.4

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Preparation
Ensure your dataset follows this structure:
```
data/
  base_training/
    images/          # 1024x1024 PNG/JPG images
    labelTxt/        # DOTA format .txt annotation files
```

**Dataset Requirements:**
- Images: 1024×1024 pixels (cropped from DOTA)
- Annotations: DOTA format .txt files
- Classes: 12 base classes (bridge, ground-track-field, small-vehicle, large-vehicle, ship, basketball-court, storage-tank, soccer-ball-field, roundabout, harbor, swimming-pool, helicopter)

### Step 3: Start Training
```bash
# Activate environment
conda activate s2anet

# Start training
python tools/train.py configs/s2anet_r50_fpn_1x_dota12_base.py --work_dir work_dirs/s2anet_base12
```

### Step 4: Monitor Training
- **Logs**: Check `work_dirs/s2anet_base12/` for training logs
- **Expected duration**: ~9-10 hours for 12 epochs
- **Checkpoints**: Saved automatically every epoch

## 📊 Training Monitoring

### What to Expect
- **Early iterations**: May see loss spikes (normal for S2A-Net)
- **After ~1000 iterations**: Losses stabilize around 4.0-4.6
- **Gradient norms**: Should be 2-3 (healthy range)
- **Memory usage**: ~3GB GPU memory

### Key Metrics
- **FAM losses**: `fam.loss_cls`, `fam.loss_bbox` (Feature Alignment Module)
- **ODM losses**: `odm.loss_cls`, `odm.loss_bbox` (Oriented Detection Module)
- **Total loss**: Combined loss from both heads
- **Learning rate**: Starts at 0.01, drops at epochs 8 & 11

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **Training Stuck at Start**
```bash
# Use Windows-friendly settings
python tools/train.py configs/s2anet_r50_fpn_1x_dota12_base.py --work_dir work_dirs/s2anet_base12 --cfg-options data.workers_per_gpu=0 data.samples_per_gpu=2
```

#### 2. **CUDA Out of Memory**
```bash
# Reduce batch size
python tools/train.py configs/s2anet_r50_fpn_1x_dota12_base.py --work_dir work_dirs/s2anet_base12 --cfg-options data.samples_per_gpu=1
```

#### 3. **Import Errors**
```bash
# Ensure MMRotate is imported
python -c "import mmrotate; print('MMRotate imported successfully')"
```

#### 4. **Dataset Loading Issues**
```bash
# Test dataset loading
python -c "from mmcv import Config; import mmrotate; from mmdet.datasets import build_dataset; cfg = Config.fromfile('configs/s2anet_r50_fpn_1x_dota12_base.py'); ds = build_dataset(cfg.data.train); print('Dataset loaded:', len(ds), 'samples')"
```

## 📁 File Structure

```
s2anet/
├── configs/
│   └── s2anet_r50_fpn_1x_dota12_base.py  # 12-class training config
├── tools/
│   └── train.py                          # Updated training script
├── data/
│   └── base_training/                    # Your 12-class dataset
│       ├── images/                       # 1024x1024 images
│       └── labelTxt/                     # DOTA annotations
├── work_dirs/
│   └── s2anet_base12/                    # Training outputs
└── S2ANET_TRAINING_GUIDE.md             # This guide
```


