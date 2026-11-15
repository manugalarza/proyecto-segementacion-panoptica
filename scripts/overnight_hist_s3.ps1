# Overnight pipeline for the leak-free historical S3 seed 0 split (Jorge's partition).
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\overnight_hist_s3.ps1
# Every step is skipped if its output already exists; failures are logged and the
# pipeline continues. Log: ..\logs_hist_s3_overnight.txt

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_hist_s3_overnight.txt"
Start-Transcript -Path $Log -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension (admin?)." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$TestHud = "..\test_hud_pareado_s0"
$GtTest = "..\gt_test_hud_s0"
$CkA = "..\checkpoints_hist_s3\A_full"
$CkI = "..\checkpoints_hist_s3\A_inst2"
$InstTrain = "..\sam_hist_s3_instances\train"
$Res = "..\resultados_hist_s3"

# 1. Pseudo ground truth for the paired test (SAM from the test boxes, HUD version labels).
Step "SAM pseudo-GT test (things + stuff from boxes)" @("scripts\sam_from_boxes.py",
    "--images", "$TestHud\con_hud\images", "--labels", "test=$TestHud\con_hud\labels",
    "--out", "..\sam_test_hud_panels", "--save-masks", $GtTest, "--no-panels",
    "--model", "sam_b.pt", "--device", "0")

# 2. Evaluate A_full (historical S3) with and without HUD.
if (Test-Path "$CkA\epoch_150.pt") {
    Step "Eval A_full con HUD" @("scripts\eval_full.py", "--images", "$TestHud\con_hud\images", "--gt", $GtTest,
        "--checkpoint", "hist_A_full=$CkA\epoch_150.pt", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\A_full_con_hud")
    Step "Eval A_full sin HUD" @("scripts\eval_full.py", "--images", "$TestHud\sin_hud\images", "--gt", $GtTest,
        "--checkpoint", "hist_A_full=$CkA\epoch_150.pt", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\A_full_sin_hud")
} else { Write-Host "A_full epoch_150 no existe; se omite la evaluacion" }

# 3. SAM instance masks for the S3 train split (needed only for A_inst2).
if (-not (Test-Path "$InstTrain")) {
    Step "SAM instance masks train S3" @("scripts\sam_from_boxes.py",
        "--images", "$Data\train\images", "--labels", "train=$Data\train\labels",
        "--out", "..\sam_hist_s3_panels", "--save-instances", $InstTrain, "--no-panels",
        "--model", "sam_b.pt", "--device", "0")
} else { Write-Host "Instancias SAM train ya existen; se omiten." }

# 4. Train A_inst2 on the same split.
if (-not (Test-Path "$CkI\epoch_150.pt")) {
    Step "Train A_inst2 S3" @("scripts\train_gt_variant.py", "--data", $Data, "--out", $CkI,
        "--encoder", "resnet18", "--epochs", "150", "--seed", "0", "--device", "cuda", "--instances", $InstTrain)
} else { Write-Host "A_inst2 epoch_150 ya existe; se omite." }

# 5. Evaluate A_inst2 with and without HUD.
if (Test-Path "$CkI\epoch_150.pt") {
    Step "Eval A_inst2 con HUD" @("scripts\eval_full.py", "--images", "$TestHud\con_hud\images", "--gt", $GtTest,
        "--checkpoint", "hist_A_inst2=$CkI\epoch_150.pt", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\A_inst2_con_hud")
    Step "Eval A_inst2 sin HUD" @("scripts\eval_full.py", "--images", "$TestHud\sin_hud\images", "--gt", $GtTest,
        "--checkpoint", "hist_A_inst2=$CkI\epoch_150.pt", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\A_inst2_sin_hud")
} else { Write-Host "A_inst2 epoch_150 no existe; se omite la evaluacion" }

Write-Host "=== FIN  $(Get-Date)  resultados en $Res"
Stop-Transcript | Out-Null
