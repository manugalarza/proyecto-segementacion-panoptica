# 2x2 ground-truth experiment: train Panoptic FCN with Etiquetas and with the
# April re-annotation (same images, same split), then evaluate both models
# on both test sets. Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\run_2x2.ps1
# Everything is logged to <ProjectDir>\resultados_2x2\run_2x2_log.txt.
# Steps already done are skipped (delete the folder to redo a step).

param(
    [int]$Epochs = 150,
    [string]$DataDir = "C:\Users\investigacion\Documents\OneDrive_1_7-10-2026",
    [string]$ProjectDir = "C:\Users\investigacion\Documents\Proyecto de grado"
)

$ErrorActionPreference = "Stop"
$Py = ".venv\Scripts\python.exe"
$Datos = Join-Path $ProjectDir "datos_2x2"
$Ckpt = Join-Path $ProjectDir "checkpoints_2x2"
$Res = Join-Path $ProjectDir "resultados_2x2"
$Pretrained = Join-Path $ProjectDir "epoch_63.pt"
New-Item -ItemType Directory -Force -Path $Res | Out-Null
Start-Transcript -Path (Join-Path $Res "run_2x2_log.txt") -Append

# Keep the machine awake while this runs (the 2026-09-19 overnight run died when Windows slept).
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "Could not change sleep settings (needs admin?) - make sure the machine does not sleep." }

Write-Host "=== 1/4 Build datasets $(Get-Date) ==="
if (-not (Test-Path (Join-Path $Datos "split_manifest.csv"))) {
    & $Py scripts\build_gt_datasets.py --imagenes "$DataDir\Imagenes" --etiquetas "$DataDir\Etiquetas" --abril "$DataDir\dataset_split_completo" --out $Datos
    if ($LASTEXITCODE -ne 0) { throw "build_gt_datasets failed" }
} else { Write-Host "datos_2x2 already exists, skipping." }

foreach ($Variant in @("etiquetas", "abril")) {
    $Step = if ($Variant -eq "etiquetas") { "2/4" } else { "3/4" }
    Write-Host "=== $Step Train $Variant ($Epochs epochs) $(Get-Date) ==="
    $Final = Join-Path $Ckpt "$Variant\epoch_$Epochs.pt"
    if (-not (Test-Path $Final)) {
        & $Py scripts\train_gt_variant.py --data "$Datos\$Variant" --out "$Ckpt\$Variant" --pretrained $Pretrained --epochs $Epochs
        if ($LASTEXITCODE -ne 0) { throw "training $Variant failed" }
    } else { Write-Host "$Final already exists, skipping." }
}

Write-Host "=== 4/4 Evaluate 2x2 $(Get-Date) ==="
& $Py scripts\eval_sdzi_stuff.py `
    --checkpoint "etiquetas=$Ckpt\etiquetas\epoch_$Epochs.pt" `
    --checkpoint "abril=$Ckpt\abril\epoch_$Epochs.pt" `
    --gt "etiquetas=$Datos\etiquetas" --gt "abril=$Datos\abril" `
    --out $Res
if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }

Write-Host "=== Done $(Get-Date). Results in $Res ==="
Stop-Transcript
