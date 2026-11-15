# Relative-coordinate instance kernels (CondInst-style), 2026-11-11.
# Question: does giving each instance kernel coordinates relative to its own center stop the
# predicted masks from lighting up on other, similar objects (the "merged boxes" problem), and
# does it improve building / vehicle AP on validation?
#
# Stage 1 (minutes): overfit on the same 12 train frames, focal 576x1024, with and without
#   --rel-coords (cosine LR in both). Evaluated on those frames.
# Stage 2 (~70 min per run): historical S3 seed 0, focal 576x1024, cosine LR, 150 epochs:
#     G_cos         (control: same as before + cosine LR)
#     G_cos_rc      (+ --rel-coords)
#   Evaluated on VALIDATION only, with the default boxes (largest connected part) and with
#   --box-mode full (box around the whole mask: shows whether the masks themselves stay local).
#   Test is not touched.
#
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\experimento_relcoords.ps1            # both stages
#     powershell -ExecutionPolicy Bypass -File scripts\experimento_relcoords.ps1 -Stage 1   # only stage 1
# Paste the table and the "Diagnostics" blocks printed at the end.

param([string]$Stage = "all")

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
Start-Transcript -Path "..\logs_experimento_relcoords.txt" -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$Ov = "..\overfit12"
$SamSem = "..\sam_hist_s3_semantic_train"
$SamInst = "..\sam_hist_s3_instances\train"
$Ck = "..\checkpoints_relcoords"
$Res = "..\resultados_relcoords"
New-Item -ItemType Directory -Force -Path $Ck, $Res | Out-Null
# NOTE: PowerShell variable names are case-insensitive; $Ck (folder) and the checkpoint list must differ.
$Common = @("--encoder", "resnet18", "--seed", "0", "--device", "cuda", "--masks", $SamSem, "--instances", $SamInst,
            "--heatmap-loss", "focal", "--image-size", "576", "1024", "--lr-schedule", "cosine", "--batch-size", "8")

if ($Stage -eq "all" -or $Stage -eq "1") {
    if (-not (Test-Path "$Ov\dataset.yaml")) { Write-Host "Falta ..\overfit12 (lo crea experimento_resolucion.ps1 -Stage 1)" }
    Step "Overfit O_cos" (@("scripts\train_gt_variant.py", "--data", $Ov, "--out", "$Ck\O_cos", "--epochs", "150",
        "--lr", "0.001", "--save-every", "50", "--num-workers", "0") + $Common)
    Step "Overfit O_cos_rc" (@("scripts\train_gt_variant.py", "--data", $Ov, "--out", "$Ck\O_cos_rc", "--epochs", "150",
        "--lr", "0.001", "--save-every", "50", "--num-workers", "0", "--rel-coords") + $Common)
    $CkArgs = @("--checkpoint", "O_cos=$Ck\O_cos\epoch_150.pt", "--checkpoint", "O_cos_rc=$Ck\O_cos_rc\epoch_150.pt")
    Step "Eval overfit (cajas por defecto)" (@("scripts\eval_detection.py", "--data", $Ov, "--split", "train",
        "--device", "cuda", "--bootstrap", "200", "--out", "$Res\etapa1_overfit12") + $CkArgs)
    Step "Eval overfit (box-mode full)" (@("scripts\eval_detection.py", "--data", $Ov, "--split", "train", "--box-mode", "full",
        "--device", "cuda", "--bootstrap", "200", "--out", "$Res\etapa1_overfit12_full") + $CkArgs)
}

if ($Stage -eq "all" -or $Stage -eq "2") {
    Step "Entrenar G_cos" (@("scripts\train_gt_variant.py", "--data", $Data, "--out", "$Ck\G_cos", "--epochs", "150",
        "--lr", "0.0003") + $Common)
    Step "Entrenar G_cos_rc" (@("scripts\train_gt_variant.py", "--data", $Data, "--out", "$Ck\G_cos_rc", "--epochs", "150",
        "--lr", "0.0003", "--rel-coords") + $Common)
    $CkArgs = @("--checkpoint", "F576_e130_ref=..\checkpoints_resolucion\F_focal_576x1024\epoch_130.pt")
    foreach ($run in @("G_cos", "G_cos_rc")) {
        foreach ($e in @(130, 140, 150)) { $CkArgs += @("--checkpoint", "$($run)_e$e=$Ck\$run\epoch_$e.pt") }
    }
    Step "Eval val (cajas por defecto)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val",
        "--device", "cuda", "--bootstrap", "500", "--out", "$Res\etapa2_val") + $CkArgs)
    Step "Eval val (box-mode full)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val", "--box-mode", "full",
        "--device", "cuda", "--bootstrap", "200", "--out", "$Res\etapa2_val_full") + $CkArgs)
}

Write-Host "`n=== TABLA PARA PEGAR EN EL CHAT ==="
& $Py scripts\summarize_detection.py $Res | Tee-Object -FilePath "$Res\tabla.txt"
Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
