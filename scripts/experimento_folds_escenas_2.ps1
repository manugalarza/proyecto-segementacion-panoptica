# EXPLORATORY, second round on the v0 scene folds (2026-11-11). Needs experimento_folds_escenas.ps1 done
# (datasets ..\datos_escenas_v0_f0/f1 and their SAM masks).
#   A. YOLO reference (yolo11m, 1024 px, 60 epochs, Ultralytics default augmentation) on the same
#      train/val, scored on val and test with eval_detection.py: is the gap the model or the data?
#   B. Our best model + --augment (flips, zoom-in crop, color jitter): does augmentation help on unseen scenes?
# Test of each fold is evaluated once per model (final epoch / YOLO best.pt chosen on its own val).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\experimento_folds_escenas_2.ps1
# Paste the final summary.

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
Start-Transcript -Path "..\logs_experimento_folds_escenas_2.txt" -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

foreach ($k in @(0, 1)) {
    $Data = "..\datos_escenas_v0_f$k"
    $Res = "..\resultados_escenas_v0_f$($k)_ronda2"
    if (-not (Test-Path "$Data\dataset.yaml")) { Write-Host "Falta $Data (correr experimento_folds_escenas.ps1)"; continue }

    # A. YOLO
    $Y = "..\yolo_escenas_v0_f$k"
    Step "YOLO fold $k" @("scripts\yolo_reference.py", "--data", $Data, "--out", $Y)
    foreach ($split in @("val", "test")) {
        Step "Eval YOLO $split fold $k" @("scripts\eval_detection.py", "--data", $Data, "--split", $split,
            "--predictions", "YOLO11m=$Y\pred_$split\labels", "--bootstrap", "1000", "--out", "$Res\$($split)_yolo")
    }

    # B. Our model + augmentation
    $Ck = "..\checkpoints_escenas_v0_f$k\focal576_cos_rc_aug"
    if (-not (Test-Path "$Ck\epoch_150.pt")) {
        Step "Entrenar aug fold $k" @("scripts\train_gt_variant.py", "--data", $Data, "--out", $Ck, "--encoder", "resnet18",
            "--epochs", "150", "--batch-size", "8", "--seed", "0", "--device", "cuda",
            "--masks", "..\sam_escenas_v0_f$($k)_semantic_train", "--instances", "..\sam_escenas_v0_f$($k)_instances_train",
            "--lr", "0.0003", "--heatmap-loss", "focal", "--image-size", "576", "1024", "--lr-schedule", "cosine",
            "--rel-coords", "--augment")
    }
    $CkArgs = @("--checkpoint", "rc_aug_e150=$Ck\epoch_150.pt",
                "--checkpoint", "rc_noaug_e150=..\checkpoints_escenas_v0_f$k\focal576_cos_rc\epoch_150.pt")
    foreach ($split in @("val", "test")) {
        Step "Eval aug $split fold $k" (@("scripts\eval_detection.py", "--data", $Data, "--split", $split, "--device", "cuda",
            "--bootstrap", "1000", "--out", "$Res\$($split)_ours") + $CkArgs)
        Step "Eval aug $split fold $k (full)" (@("scripts\eval_detection.py", "--data", $Data, "--split", $split, "--device", "cuda",
            "--box-mode", "full", "--bootstrap", "1000", "--out", "$Res\$($split)_ours_full") + $CkArgs)
    }
}

Write-Host "`n=== RESUMEN PARA PEGAR EN EL CHAT ==="
foreach ($k in @(0, 1)) {
    Write-Host "--- fold $k como test ---"
    & $Py scripts\summarize_detection.py "..\resultados_escenas_v0_f$($k)_ronda2"
}
Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
