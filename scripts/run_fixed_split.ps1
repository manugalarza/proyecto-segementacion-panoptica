# Full pipeline for a partition given as an assignment CSV (content_id,split), e.g. Jorge's
# fixed_test_train_val_v1 (2026-11-11). Splits other than train/val/test (e.g. quarantine) are
# listed in the manifest and not copied.
#   1. Build the dataset: labels from datos_v2 when the frame is there, else the original
#      Etiquetas (SDZI kept as class 4, ignored), plus the reviewed machinery boxes; one
#      record per content_id (exact duplicates dropped).
#   2. SAM pseudo-masks (stuff target + instance masks) for its train split.
#   3. Train (focal heatmap, 576x1024, cosine LR, 150 epochs; -RelCoords adds relative coordinates).
#   4. Evaluate epochs 130/140/150 on VALIDATION. Test only with -EvalTest (epoch 150).
#
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\run_fixed_split.ps1 -Assignment "..\ruta\al\csv_de_jorge.csv" -Name fixed_v1 [-RelCoords]
# Steps whose output already exists are skipped, so it can be re-run after a failure.

param(
    [Parameter(Mandatory = $true)][string]$Assignment,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Partition = "..\FLIR_splits_para_revision\expanded-S3-frames\expanded-S3-seed0-frames.csv",
    [switch]$RelCoords,
    [switch]$EvalTest,
    [switch]$OriginalLabels,
    [int]$Batch = 8
)

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
Start-Transcript -Path "..\logs_run_$Name.txt" -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO: $name exit=$LASTEXITCODE"; return $false }
    Write-Host "OK: $name"; return $true
}

# 0. The assignment must have content_id and split columns.
$hdr = (Get-Content $Assignment -TotalCount 1)
if ($hdr -notmatch "content_id" -or $hdr -notmatch "split") {
    Write-Host "El CSV de asignacion debe tener columnas content_id y split. Encabezado: $hdr"
    Stop-Transcript | Out-Null; exit 1
}

$Data = "..\datos_$Name"
$SamSem = "..\sam_$($Name)_semantic_train"
$SamInst = "..\sam_$($Name)_instances_train"
$Run = if ($RelCoords) { "focal576_cos_rc" } else { "focal576_cos" }
$Ck = "..\checkpoints_$Name\$Run"
$Res = "..\resultados_$Name"

# 1. Dataset
if (-not (Test-Path "$Data\dataset.yaml")) {
    $build = @("scripts\build_split_from_csv.py", "--partition-csv", $Partition,
        "--assignment", $Assignment, "--images-root", "..\original\Imagenes", "--labels-root", "..\original\Etiquetas",
        "--unique-content", "--out", $Data)
    # Scenario A (default): datos_v2 labels when available + reviewed machinery boxes.
    # Scenario B (-OriginalLabels): original FAC Etiquetas only (no machinery class), as Jorge's YOLO.
    if (-not $OriginalLabels) { $build += @("--datos-v2", "..\datos_v2", "--extra-labels", "configs\labels\etiquetas_machinery_review_2026-11-10.csv") }
    $ok = Step "Construir dataset $Name" $build
    if (-not $ok) { Stop-Transcript | Out-Null; exit 1 }
} else { Write-Host "Dataset ya existe: $Data" }

# 2. SAM pseudo-masks for train
if (-not (Test-Path "$SamInst")) {
    $ok = Step "SAM train" @("scripts\sam_from_boxes.py", "--images", "$Data\train\images", "--labels", "train=$Data\train\labels",
        "--out", "..\sam_$($Name)_panels", "--save-masks", $SamSem, "--save-instances", $SamInst,
        "--no-panels", "--model", "sam_b.pt", "--device", "0")
    if (-not $ok) { Stop-Transcript | Out-Null; exit 1 }
} else { Write-Host "SAM train ya existe." }

# 3. Train
$train = @("scripts\train_gt_variant.py", "--data", $Data, "--out", $Ck, "--encoder", "resnet18", "--epochs", "150",
    "--batch-size", "$Batch", "--seed", "0", "--device", "cuda", "--masks", $SamSem, "--instances", $SamInst,
    "--lr", "0.0003", "--heatmap-loss", "focal", "--image-size", "576", "1024", "--lr-schedule", "cosine")
if ($RelCoords) { $train += "--rel-coords" }
if (-not (Test-Path "$Ck\epoch_150.pt")) { $null = Step "Entrenar $Run" $train } else { Write-Host "Ya entrenado: $Ck" }

# 4. Validation
$CkArgs = @()
foreach ($e in @(130, 140, 150)) { if (Test-Path "$Ck\epoch_$e.pt") { $CkArgs += @("--checkpoint", "$($Run)_e$e=$Ck\epoch_$e.pt") } }
$null = Step "Eval val (largest_cc)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val", "--device", "cuda",
    "--bootstrap", "1000", "--out", "$Res\val_$Run") + $CkArgs)
$null = Step "Eval val (box-mode full)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val", "--device", "cuda",
    "--box-mode", "full", "--bootstrap", "1000", "--out", "$Res\val_$($Run)_full") + $CkArgs)

# 5. Test, only when asked (-EvalTest): for exploratory protocols whose test fold is meant to be
#    evaluated (e.g. the scene folds). Final epoch only (cosine LR: no checkpoint choice on test).
if ($EvalTest -and (Test-Path "$Ck\epoch_150.pt")) {
    $T = @("--checkpoint", "$($Run)_e150=$Ck\epoch_150.pt")
    $null = Step "Eval TEST (largest_cc)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "test", "--device", "cuda",
        "--bootstrap", "1000", "--out", "$Res\test_$Run") + $T)
    $null = Step "Eval TEST (box-mode full)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "test", "--device", "cuda",
        "--box-mode", "full", "--bootstrap", "1000", "--out", "$Res\test_$($Run)_full") + $T)
}

Write-Host "`n=== TABLA PARA PEGAR EN EL CHAT ==="
& $Py scripts\summarize_detection.py $Res | Tee-Object -FilePath "$Res\tabla.txt"
Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
