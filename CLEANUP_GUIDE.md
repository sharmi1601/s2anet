# 🧹 S2ANet Project Cleanup Guide

This guide helps you clean up your S2ANet project to avoid import conflicts and maintain a lean training environment.

## ⚠️ Why Clean Up?

Your current project has **local source copies** of `mmdet_legacy/`, `mmcv/`, `mmrotate/` etc. that can **shadow** your pip-installed packages, causing mysterious import errors. For clean training with pip-installed MMDetection + MMRotate, you only need:

- Your training script (`tools/train.py`)
- Your config (`configs/s2anet_r50_fpn_1x_dota12_base.py`)
- Your dataset (`data/`)
- Essential utilities (`DOTA_devkit/`, `checkpoints/`)

## 🚀 Quick Cleanup (Recommended)

### 1. **Dry Run First** (Safe Preview)
```powershell
# Open PowerShell in your project root
.\cleanup_s2anet.ps1
```
This shows what **would** be moved without actually doing it.

### 2. **Run Actual Cleanup**
```powershell
.\cleanup_s2anet.ps1 -DryRun:$false
```

### 3. **Verify Clean Environment**
```bash
python verify_clean_environment.py
```

## 📋 What Gets Archived

The script **safely moves** (doesn't delete) these to `_archive_YYYYMMDD_HHMMSS/`:

### 🔥 **Critical: Local Package Sources**
- `mmdet_legacy/` - Legacy MMDetection source
- `mmdet/`, `mmcv/`, `mmrotate/` - Any local source trees
- `*.egg-info/` - Old editable install artifacts

### 📁 **Unused Config Directories** 
- `configs/albu_example/`, `configs/atss/`, `configs/cityscapes/` 
- `configs/dcn/`, `configs/fcos/`, `configs/hrnet/`, etc.
- Most individual `.py` configs (RetinaNet, Faster R-CNN, etc.)

### 📊 **Results & Build Artifacts**
- `results_before_nms/`, `results_after_nms/`, `results.pkl`
- `build/`, `dist/`, `.eggs/`
- `work_dirs/s2anet_base_training/` (old logs)

### 🐳 **Optional Components**
- `docker/` (if not using Docker)
- `test_setup.py`, `test_training_setup.py`

## ✅ What Stays (Essential Files)

```
s2anet/
├── configs/
│   ├── s2anet_r50_fpn_1x_dota12_base.py  ✅ Your 12-class config
│   ├── dota/                              ✅ DOTA-specific configs  
│   └── rotated_iou/                       ✅ IoU loss variants
├── tools/
│   └── train.py                           ✅ Training script
├── data/                                  ✅ Your datasets
├── work_dirs/s2anet_base12/              ✅ Current training outputs
├── checkpoints/                           ✅ Pre-trained models
├── DOTA_devkit/                          ✅ Dataset utilities
├── demo/                                  ✅ Demo files
├── docs/                                  ✅ Documentation
├── requirements.txt                       ✅ Dependencies
├── setup_environment.bat                 ✅ Setup script
└── README.md                             ✅ Main readme
```

## 🔍 Post-Cleanup Verification

After cleanup, run:

```bash
python verify_clean_environment.py
```

**Expected output:**
```
✓ mmdet: /path/to/conda/envs/s2anet/lib/python3.8/site-packages/mmdet/__init__.py
✓ mmcv: /path/to/conda/envs/s2anet/lib/python3.8/site-packages/mmcv/__init__.py  
✓ mmrotate: /path/to/conda/envs/s2anet/lib/python3.8/site-packages/mmrotate/__init__.py

🎉 Environment is clean and ready for S2ANet training!
```

## 🆘 Troubleshooting

### ❌ **Still seeing local imports?**
```bash
# Check for any remaining local packages
ls -la | grep -E "(mmdet|mmcv|mmrotate)"

# If found, manually rename them
mv mmdet_legacy mmdet_legacy_DISABLED
```

### ❌ **Import errors after cleanup?**
```bash
# Reinstall packages to ensure clean state
pip uninstall mmdet mmrotate mmcv-full
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.28.2 mmrotate==0.3.4
```

### ❌ **Need to restore something?**
Everything is safely archived in `_archive_YYYYMMDD_HHMMSS/`. You can copy back any needed files.

## 🎯 Training After Cleanup

```bash
# Activate environment
conda activate s2anet

# Start training (should work cleanly now)
python tools/train.py configs/s2anet_r50_fpn_1x_dota12_base.py --work_dir work_dirs/s2anet_base12
```

## 💾 Disk Space Saved

Typical cleanup saves **2-5 GB** by removing:
- Redundant config files
- Old build artifacts  
- Duplicate source trees
- Result caches

---

**Pro Tip:** Keep this cleaned-up version as your **training environment**. If you later need to modify S2ANet source code, create a separate **development environment** to avoid conflicts.
