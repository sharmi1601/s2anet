# S2A-Net Training Setup Guide

This guide helps you set up S2A-Net for training with 12 base classes on RTX 4070.

## Prerequisites

- **GPU**: RTX 4070 (or compatible GPU with compute capability 8.9+)
- **CUDA**: Version 12.x (tested with CUDA 12.1+)
- **Python**: 3.8-3.10
- **PyTorch**: 1.12+ with CUDA support

## Installation Steps

### 1. Environment Setup

```bash
# Create conda environment
conda create -n s2anet python=3.9 -y
conda activate s2anet

# Install PyTorch with CUDA 12.x support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
conda install swig -y
```

### 2. Install MMCV

```bash
# Install compatible MMCV version
pip install mmcv-full>=1.0.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch1.12/index.html
```

### 3. Install S2A-Net

```bash
# Clone and setup S2A-Net
git clone <your-s2anet-repo>
cd s2anet

# Install dependencies
pip install -r requirements.txt

# Set environment variables for RTX 4070
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9+PTX"
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# Compile S2A-Net
python setup.py develop
```

### 4. Compile DOTA Devkit

```bash
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace
```

### 5. Test Installation

```bash
# Run the test script
python test_training_setup.py
```

## Training Configuration

### Key Changes Made for 12 Classes:

1. **Model Configuration**:
   - `num_classes=13` (12 base classes + background)
   - Anchor scales: `[4]` (produces {32², 64², 128², 256², 512²} with strides)
   - ORConv enabled for rotation invariance

2. **Training Parameters**:
   - Learning rate: 0.01
   - Weight decay: 1e-4
   - Batch size: 2 per GPU (adjust based on memory)
   - Epochs: 12
   - LR drops at epochs 8 and 11

3. **Data Augmentation**:
   - Random flip + random rotation
   - Image size: 1024×1024
   - Stride 824 crops

## Starting Training

### For 12 Base Classes:

```bash
python tools/train.py configs/dota/s2anet_r50_fpn_1x_dota_12_classes.py
```

### For Multi-GPU Training:

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/dota/s2anet_r50_fpn_1x_dota_12_classes.py --launcher pytorch
```

## Monitoring Training

- **Logs**: Check `work_dirs/s2anet_r50_fpn_1x_dota_12_classes/` for training logs
- **Checkpoints**: Saved every epoch in `work_dirs/s2anet_r50_fpn_1x_dota_12_classes/`
- **TensorBoard**: Add TensorBoard hook to config for visualization

## Troubleshooting

### Common Issues:

1. **CUDA Error**: "no kernel image is available"
   - Solution: Ensure `TORCH_CUDA_ARCH_LIST` includes "8.9+PTX"

2. **MMCV Import Error**:
   - Solution: Install compatible MMCV version for your PyTorch version

3. **Memory Error**:
   - Solution: Reduce `imgs_per_gpu` in config or use gradient accumulation

4. **Compilation Error**:
   - Solution: Ensure CUDA toolkit and PyTorch versions are compatible

### Verification:

Run the test script to verify everything works:
```bash
python test_training_setup.py
```

## Expected Results

- **Training Time**: ~12-24 hours on RTX 4070 (depending on dataset size)
- **Memory Usage**: ~8-12GB GPU memory
- **Convergence**: Should see decreasing loss after epoch 3-4

## Next Steps

After base training is complete:
1. Save the base model checkpoint
2. Prepare few-shot datasets with novel classes
3. Implement few-shot fine-tuning pipeline
4. Evaluate on test set

## Support

If you encounter issues:
1. Check the test script output
2. Verify CUDA and PyTorch compatibility
3. Ensure all dependencies are correctly installed
4. Check GPU memory availability

