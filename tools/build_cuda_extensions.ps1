param(
    [string]$VcVars = '',
    [string]$Toolset = '14.44',
    [string]$CudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
    [string]$CudaArch = '12.0',
    [int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create the repository .venv first.' }
if (!$VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { throw 'Pass -VcVars with the path to vcvarsall.bat.' }
    $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vsRoot) { throw 'Visual Studio C++ tools were not found.' }
    $VcVars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvarsall.bat'
}
if (!(Test-Path -LiteralPath $VcVars)) { throw "Missing compiler setup: $VcVars" }
if (!(Test-Path -LiteralPath "$CudaPath\bin\nvcc.exe")) { throw "Missing CUDA toolkit: $CudaPath" }
$env:VSLANG = '1033'
& cmd.exe /d /c "call `"$VcVars`" x64 -vcvars_ver=$Toolset >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment initialization failed.' }
# Match the Chinese Windows OEM encoding expected by PyTorch's compiler probe.
& cmd.exe /d /c 'chcp 936 >nul'
if ($LASTEXITCODE -ne 0) { throw 'Use a normal PowerShell console with an active code page.' }
$env:DISTUTILS_USE_SDK = '1'
$env:CUDA_HOME = $CudaPath
$env:TORCH_CUDA_ARCH_LIST = $CudaArch
$env:MAX_JOBS = "$Jobs"
$env:PATH = "$(Split-Path $python -Parent);$CudaPath\bin;$env:PATH"
$setup = Join-Path $repo '.local-setup'
New-Item -ItemType Directory -Force -Path $setup | Out-Null
Push-Location -LiteralPath $repo
try {
    & $python -I tools/patch_torch_header.py
    if ($LASTEXITCODE -ne 0) { throw 'Header verification/patch failed.' }
    & $python -I -c 'import json,sys,torch; from pathlib import Path; p=Path(sys.prefix); assert Path(torch.__file__).resolve().is_relative_to(p.resolve()); print(json.dumps({"python":sys.executable,"torch":torch.__version__,"torch_file":torch.__file__,"cuda":torch.version.cuda},indent=2))' > "$setup\build-environment.json"
    if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
    foreach ($extension in @('diff-gaussian-rasterization', 'simple-knn', 'fused-ssim')) {
        $source = Join-Path $repo "submodules\$extension"
        Write-Output "Building $extension with $python"
        & $python -I -m pip install --no-build-isolation --no-deps --force-reinstall --no-cache-dir $source --log "$setup\$extension-own-venv.log"
        if ($LASTEXITCODE -ne 0) { throw "Failed: $extension. See .local-setup/$extension-own-venv.log" }
    }
    & $python -I tools/verify_cuda_extensions.py
    if ($LASTEXITCODE -ne 0) { throw 'CUDA numerical/gradient checks failed.' }
    & $python -I -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
} finally {
    Pop-Location
}
