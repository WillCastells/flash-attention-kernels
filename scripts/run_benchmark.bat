@echo off
cd /d "%~dp0\.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set TORCH_CUDA_ARCH_LIST=7.5
python benchmarks\benchmark.py
