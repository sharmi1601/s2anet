@echo off
REM Setup script for S2A-Net training environment on Windows

echo Setting up S2A-Net training environment...

REM Activate conda environment
echo Activating s2anet conda environment...
call conda activate s2anet

REM Check if environment is activated
if "%CONDA_DEFAULT_ENV%" neq "s2anet" (
    echo Error: s2anet environment not activated. Please run: conda activate s2anet
    pause
    exit /b 1
)

echo Environment activated: %CONDA_DEFAULT_ENV%

REM Install MMCV
echo Installing MMCV...
pip install mmcv-full>=1.0.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch1.12/index.html

REM Install other requirements
echo Installing requirements...
pip install -r requirements.txt

REM Set environment variables for RTX 4070
echo Setting CUDA environment variables...
set TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6;8.9+PTX
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set FORCE_CUDA=1
set MMCV_WITH_OPS=1

REM Compile S2A-Net
echo Compiling S2A-Net...
python setup.py develop

REM Compile DOTA devkit
echo Compiling DOTA devkit...
cd DOTA_devkit\polyiou
swig -c++ -python csrc\polyiou.i
python setup.py build_ext --inplace
cd ..\..

REM Test installation
echo Testing installation...
python test_training_setup.py

echo Setup complete! You can now start training with:
echo python tools\train.py configs\dota\s2anet_r50_fpn_1x_dota_12_classes.py
pause

