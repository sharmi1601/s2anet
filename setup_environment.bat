@echo off
echo ========================================
echo S2A-Net Training Environment Setup
echo ========================================

echo Creating conda environment...
conda create -n s2anet python=3.8 -y

echo Activating environment...
call conda activate s2anet

echo Installing PyTorch...
conda install pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0 cudatoolkit=11.3 -c pytorch -y

echo Installing MMCV...
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

echo Installing MMDetection...
pip install mmdet==2.28.2

echo Installing MMRotate...
pip install mmrotate==0.3.4

echo Installing other dependencies...
pip install -r requirements.txt

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start training:
echo 1. conda activate s2anet
echo 2. python tools/train.py configs/s2anet_r50_fpn_1x_dota12_base.py --work_dir work_dirs/s2anet_base12
echo.
echo See S2ANET_TRAINING_GUIDE.md for detailed instructions.
pause

