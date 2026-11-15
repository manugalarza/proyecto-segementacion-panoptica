# Resolution / aspect-ratio experiment with the focal heatmap loss (2026-11-10).
#
# Stage 1 (overfit, ~minutes per run): the same 12 train frames, 150 epochs, evaluated on
#   those same frames. Tells whether the architecture can fit them at each input size.
#     O_bce_512        original loss, 512x512 (reference)
#     O_focal_512      focal loss, 512x512
#     O_focal_384x672  focal, ~same pixel count as 512x512 but 16:9 (aspect only)
#     O_focal_576x1024 focal, 16:9 and ~2.25x the pixels (aspect + resolution)
# Stage 2 (full training, ~1-2 h per run): historical S3 seed 0 with the corrected labels,
#   same settings as C (SAM stuff masks + SAM instances, lr 3e-4, 150 epochs). Evaluated on
#   VALIDATION only (test is not touched), next to the existing C checkpoint.
#     F_focal_512, F_focal_576x1024
#
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\experimento_resolucion.ps1            # both stages
#     powershell -ExecutionPolicy Bypass -File scripts\experimento_resolucion.ps1 -Stage 1   # only stage 1
# If a 576x1024 run fails with "CUDA out of memory", rerun with -BatchHi 4.
# At the end it prints one table: paste that table (or ..\resultados_resolucion\tabla.txt) in the chat.
# Log: ..\logs_experimento_resolucion.txt

param(
    [string]$Stage = "all",
    [int]$Batch = 8,
    [int]$BatchHi = 8
)

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_experimento_resolucion.txt"
Start-Transcript -Path $Log -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$SamSem = "..\sam_hist_s3_semantic_train"
$SamInst = "..\sam_hist_s3_instances\train"
$Ck = "..\checkpoints_resolucion"
$Res = "..\resultados_resolucion"
New-Item -ItemType Directory -Force -Path $Ck, $Res | Out-Null

# 0. Refresh the SAM masks of the 6 train label files changed by the machinery review
#    (SDZI zones back to class 4, tight machinery boxes added). Only these files are rewritten.
$Changed = @("video_11min_116__r1553", "video_11min_279__r0588", "video_11min_458__r0727",
             "video_11min_459__r0728", "video_11min_601__r0865", "video_11min_601__r0866")
$Fix = "..\_tmp_sam_fix"
if (-not (Test-Path "$Fix\done.txt")) {
    New-Item -ItemType Directory -Force -Path "$Fix\images", "$Fix\labels" | Out-Null
    foreach ($s in $Changed) {
        Copy-Item "$Data\train\images\$s.jpg" "$Fix\images\" -Force
        Copy-Item "$Data\train\labels\$s.txt" "$Fix\labels\" -Force
    }
    Step "SAM refresh (6 frames revisados)" @("scripts\sam_from_boxes.py", "--images", "$Fix\images", "--labels", "train=$Fix\labels",
        "--out", "$Fix\panels", "--save-masks", $SamSem, "--save-instances", $SamInst, "--no-panels", "--model", "sam_b.pt", "--device", "0")
    if ($LASTEXITCODE -eq 0) { "ok" | Out-File "$Fix\done.txt" }
}

# Stage 1: overfit on 12 train frames (no DataLoader workers: with 12 frames, starting
# 4 Windows worker processes that each import torch costs more than it saves)
if ($Stage -eq "all" -or $Stage -eq "1") {
    $Ov = "..\overfit12"
    $Frames = @("video_11min_112__r0456", "video_13min_243__r1184", "video_13min_646__r1451", "video_13min_594__r1415",
                "video_13min_526__r1362", "video_13min_011__r0237", "video_11min_471__r0739", "video_13min_665__r0355",
                "video_13min_147__r1109", "video_13min_523__r1360", "video_13min_088__r1052", "video_13min_125__r1087")
    New-Item -ItemType Directory -Force -Path "$Ov\train\images", "$Ov\train\labels" | Out-Null
    foreach ($s in $Frames) {
        Copy-Item "$Data\train\images\$s.jpg" "$Ov\train\images\" -Force
        Copy-Item "$Data\train\labels\$s.txt" "$Ov\train\labels\" -Force
    }
    Copy-Item "$Data\dataset.yaml" "$Ov\dataset.yaml" -Force

    $runs = @(
        @{ name = "O_bce_512";        loss = "bce";   h = "512"; w = "512";  b = $Batch },
        @{ name = "O_focal_512";      loss = "focal"; h = "512"; w = "512";  b = $Batch },
        @{ name = "O_focal_384x672";  loss = "focal"; h = "384"; w = "672";  b = $Batch },
        @{ name = "O_focal_576x1024"; loss = "focal"; h = "576"; w = "1024"; b = $BatchHi }
    )
    foreach ($r in $runs) {
        Step ("Overfit " + $r.name) @("scripts\train_gt_variant.py", "--data", $Ov, "--out", "$Ck\$($r.name)",
            "--encoder", "resnet18", "--epochs", "150", "--batch-size", "$($r.b)", "--seed", "0", "--device", "cuda",
            "--masks", $SamSem, "--instances", $SamInst, "--lr", "0.001", "--save-every", "50",
            "--heatmap-loss", $r.loss, "--image-size", $r.h, $r.w, "--num-workers", "0")
    }
    $ckargs = @()
    foreach ($r in $runs) { $ckargs += @("--checkpoint", "$($r.name)=$Ck\$($r.name)\epoch_150.pt") }
    Step "Eval overfit (12 frames de train)" (@("scripts\eval_detection.py", "--data", $Ov, "--split", "train",
        "--device", "cuda", "--bootstrap", "200", "--out", "$Res\etapa1_overfit12") + $ckargs)
}

# Stage 2: full training, validation only
if ($Stage -eq "all" -or $Stage -eq "2") {
    $full = @(
        @{ name = "F_focal_512";      h = "512"; w = "512";  b = $Batch },
        @{ name = "F_focal_576x1024"; h = "576"; w = "1024"; b = $BatchHi }
    )
    foreach ($r in $full) {
        Step ("Entrenar " + $r.name) @("scripts\train_gt_variant.py", "--data", $Data, "--out", "$Ck\$($r.name)",
            "--encoder", "resnet18", "--epochs", "150", "--batch-size", "$($r.b)", "--seed", "0", "--device", "cuda",
            "--masks", $SamSem, "--instances", $SamInst, "--lr", "0.0003",
            "--heatmap-loss", "focal", "--image-size", $r.h, $r.w)
    }
    $ckargs = @("--checkpoint", "C_bce_512=..\checkpoints_hist_s3\C_inst2_masks_lr3e4\epoch_150.pt")
    foreach ($r in $full) { $ckargs += @("--checkpoint", "$($r.name)=$Ck\$($r.name)\epoch_150.pt") }
    Step "Eval val (91 frames unicos)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val",
        "--device", "cuda", "--bootstrap", "1000", "--out", "$Res\etapa2_val") + $ckargs)
}

Write-Host "`n=== TABLA PARA PEGAR EN EL CHAT ==="
& $Py scripts\summarize_detection.py $Res | Tee-Object -FilePath "$Res\tabla.txt"
Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
