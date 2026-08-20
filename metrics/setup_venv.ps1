# 创建/更新 metrics/venv 并安装依赖（幂等，可重复执行）
# 用法（PowerShell）： .\setup_venv.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root "venv"
$Py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "==> 创建虚拟环境: $Venv"
    python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败，请确认 python 在 PATH 中" }
}

Write-Host "==> 升级 pip"
& $Py -m pip install --upgrade pip

Write-Host "==> 安装 torch / torchvision（体积较大，请耐心等待，pip 会显示进度条）"
& $Py -m pip install torch torchvision

Write-Host "==> 安装其余依赖（numpy/scikit-image/pandas/Pillow/lpips/flip_evaluator）"
& $Py -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host "==> 完成。使用方式：$Py 脚本.py ..."
