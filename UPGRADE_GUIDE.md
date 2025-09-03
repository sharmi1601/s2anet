# S2A-Net Upgrade Guide for Modern Hardware

## Overview
This guide helps you upgrade S2A-Net to work with modern GPUs like RTX 4070, RTX 4080, RTX 4090, and newer CUDA versions (12.x). The original S2A-Net was designed for older hardware and needs modifications to work with current systems.

## Prerequisites

### Hardware Requirements
- **GPU**: RTX 4070, RTX 4080, RTX 4090, or any GPU with compute capability 8.9+
- **CUDA**: Version 12.x (tested with CUDA 12.9)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space

### Software Requirements
- **Python**: 3.8-3.10
- **PyTorch**: 1.12+ with CUDA support
- **CUDA Toolkit**: 12.x
- **SWIG**: For DOTA devkit compilation

## Installation Steps

### 1. Environment Setup
```bash
# Create conda environment
conda create -n s2anet python=3.9 -y
conda activate s2anet

# Install PyTorch with CUDA 12.x support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install SWIG for DOTA devkit
conda install swig -y
```
### 2. Clone and Setup S2A-Net
```bash
# Clone the repository
git clone https://github.com/csuhan/s2anet.git
cd s2anet

# Install dependencies
pip install -r requirements.txt
```
### 3. Apply RTX 4070 Compatibility Fixes

#### A. Update setup.py
The `setup.py` file needs modification to include compute capability 8.9:

```python
# In setup.py, find the make_cuda_ext function and update:
def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}
    
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '--generate-code=arch=compute_60,code=sm_60',
            '--generate-code=arch=compute_61,code=sm_61', 
            '--generate-code=arch=compute_70,code=sm_70',
            '--generate-code=arch=compute_75,code=sm_75',
            '--generate-code=arch=compute_80,code=sm_80',
            '--generate-code=arch=compute_86,code=sm_86',
            '--generate-code=arch=compute_89,code=sm_89',  # RTX 4070
            '--generate-code=arch=compute_89,code=compute_89'  # PTX for future compatibility
        ]
        sources += sources_cuda
    else:
        raise ValueError('CUDA is not available')
    
    return extension(
        name=f'{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
```

#### B. Update CUDA Kernel Files
Several CUDA kernel files need API updates for modern PyTorch:

**Files to update:**
- `mmdet/ops/nms/src/nms_cuda.cpp`
- `mmdet/ops/roi_pool/src/roi_pool_cuda.cpp`
- `mmdet/ops/roi_align/src/roi_align_cuda.cpp`
- `mmdet/ops/masked_conv/src/masked_conv2d_cuda.cpp`
- `mmdet/ops/dcn/src/deform_conv_cuda.cpp`
- `mmdet/ops/dcn/src/deform_pool_cuda.cpp`
- `mmdet/ops/sigmoid_focal_loss/src/sigmoid_focal_loss_cuda.cu`
- `mmdet/ops/orn/src/cuda/ActiveRotatingFilter_cuda.cu`
- `mmdet/ops/orn/src/cuda/RotationInvariantEncoding_cuda.cu`

**Key changes needed:**
1. Replace `AT_CHECK` with `TORCH_CHECK`
2. Replace `x.type().is_cuda()` with `x.is_cuda()`
3. Replace `THC/THC.h` includes with `ATen/cuda/CUDAContext.h`
4. Replace `THCCeilDiv` with manual integer division
5. Replace `THCudaMalloc` with `cudaMalloc`
6. Replace `THCudaCheck` with `cudaGetLastError()`

### 4. Compile S2A-Net
```bash
# Set environment variables for RTX 4070
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9+PTX"
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# Compile
python setup.py develop
```

### 5. Compile DOTA Devkit
```bash
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace
```

## Testing Your Installation

### 1. Test CUDA Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
```

### 2. Test S2A-Net Imports
```python
try:
    from mmdet.ops.nms import nms_cuda
    from mmdet.ops.roi_pool import roi_pool_cuda
    from mmdet.ops.roi_align import roi_align_cuda
    print(" S2A-Net CUDA operations imported successfully!")
except ImportError as e:
    print(f" Import failed: {e}")
```


### Accuracy
- **Expected mAP**: 74.04% on DOTA dataset (as reported in paper)
- **Classes**: 15 DOTA object classes
- **Input size**: 1024x1024 pixels

##  Common Issues and Solutions

### Issue 1: "CUDA error: no kernel image is available"
**Solution**: Ensure compute capability 8.9 is included in `TORCH_CUDA_ARCH_LIST`

### Issue 2: "AT_CHECK not found"
**Solution**: Replace all `AT_CHECK` with `TORCH_CHECK` in CUDA files

### Issue 3: "THC/THC.h not found"
**Solution**: Replace THC includes with ATen includes

### Issue 4: "polyiou module import error"
**Solution**: Recompile DOTA devkit with SWIG

### Issue 5: "RuntimeError: CUDA out of memory"
**Solution**: Reduce batch size or use gradient checkpointing

##  Quick Start Commands

### Test with Pretrained Model
```bash
python tools/test.py configs/dota/s2anet_r50_fpn_1x_dota.py checkpoints/s2anet_r50_fpn_1x_dota-11c9c5f4.pth --eval bbox --out results.pkl
```



