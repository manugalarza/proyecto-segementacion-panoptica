# Overnight pipeline for the v2 taxonomy (things: vehicle/building/machinery; stuff: river/road).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\overnight_v2.ps1
# Training steps stop the pipeline on failure. Evaluation and report steps log the
# failure and continue, so the report is always generated. Log: ..\logs_overnight_v2.txt

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$R = Split-Path -Parent (Get-Location)
$Data = Join-Path $R "datos_v2"
$Ckpt = Join-Path $R "checkpoints_v2"
$GT = Join-Path $R "gt_manual"
$Res = Join-Path $R "resultados_v2"
$Rep = Join-Path $R "reporte_entrenamiento"
$Log = Join-Path $R "logs_overnight_v2.txt"
Start-Transcript -Path $Log -Append | Out-Null

try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspensión (¿admin?). Deja el equipo conectado y sin suspensión." }

function Run-Required($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO OBLIGATORIO: $name. Se detiene el pipeline."; Stop-Transcript | Out-Null; exit 1 }
}
function Run-Optional($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continúa): $name" }
}

# 1. Entrenamiento principal, semilla 0
if (-not (Test-Path (Join-Path $Ckpt "A_full\epoch_150.pt"))) {
    Run-Required "Train A_full seed 0" @("scripts\train_gt_variant.py", "--data", $Data, "--out", (Join-Path $Ckpt "A_full"),
        "--encoder", "resnet18", "--epochs", "150", "--seed", "0")
} else { Write-Host "A_full ya existe, se omite." }

# 2. Segunda semilla
if (-not (Test-Path (Join-Path $Ckpt "A_s1\epoch_150.pt"))) {
    Run-Required "Train A_s1 seed 1" @("scripts\train_gt_variant.py", "--data", $Data, "--out", (Join-Path $Ckpt "A_s1"),
        "--encoder", "resnet18", "--epochs", "150", "--seed", "1")
} else { Write-Host "A_s1 ya existe, se omite." }

$CkA = Join-Path $Ckpt "A_full\epoch_150.pt"
$CkA1 = Join-Path $Ckpt "A_s1\epoch_150.pt"
$Imgs = Join-Path $GT "images"
$Masks = Join-Path $GT "masks"

# 3. Evaluación de things (vehicle / building / machinery) sobre gt_manual
Run-Optional "Eval things" @("scripts\eval_things.py", "--images", $Imgs, "--gt", $Masks,
    "--checkpoint", "v2_A=$CkA", "--checkpoint", "v2_A_s1=$CkA1",
    "--out", (Join-Path $Res "things"), "--save-vis")

# 4. Evaluación de píxeles (river / road) + imágenes de evolución
Run-Optional "Eval pixel" @("scripts\eval_pixel.py", "--images", $Imgs, "--gt", $Masks,
    "--checkpoint", "v2_A=$CkA", "--checkpoint", "v2_A_s1=$CkA1",
    "--out", (Join-Path $Res "pixel"), "--save-vis")

# 5. Informe HTML con tiempos, épocas, curvas y las dos corridas nuevas
Run-Optional "Report" @("scripts\build_report.py",
    "--run", "v2_A_seed0=$(Join-Path $Ckpt 'A_full')", "--run", "v2_A_seed1=$(Join-Path $Ckpt 'A_s1')",
    "--run", "A_viejo=$(Join-Path $R 'runs\A')", "--run", "B_viejo=$(Join-Path $R 'runs\B')",
    "--out", $Rep)

Write-Host "=== Terminado $(Get-Date). Informe: $Rep\index.html  Resultados: $Res"
Stop-Transcript | Out-Null
