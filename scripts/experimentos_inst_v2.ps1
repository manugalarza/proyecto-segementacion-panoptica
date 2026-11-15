# Experiments to improve A_inst2 on the historical S3 seed 0 split.
#   B: stuff target from SAM semantic masks (aligned with the SAM test GT) + SAM instances, lr 1e-3.
#   C: same as B but lr 3e-4 (to reduce the loss spikes seen in A_inst2).
# Model selection uses the VALIDATION split only. The test split is evaluated once per model
# at the end, and the test numbers are reported, not used for choosing.
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\experimentos_inst_v2.ps1
# Log: ..\logs_experimentos_inst_v2.txt

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_experimentos_inst_v2.txt"
Start-Transcript -Path $Log -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$Val = "$Data\val"
$Test = "$Data\test"
$Train = "$Data\train"
$SamInst = "..\sam_hist_s3_instances\train"
$SamSemTrain = "..\sam_hist_s3_semantic_train"
$GtVal = "..\gt_val_hist_s0"
$GtTest = "..\gt_test_full_hist_s0"
$Res = "..\resultados_hist_s3\experimentos_v2"
$CkBase = "..\checkpoints_hist_s3\A_inst2\epoch_150.pt"
$CkB = "..\checkpoints_hist_s3\B_inst2_masks"
$CkC = "..\checkpoints_hist_s3\C_inst2_masks_lr3e4"

# 1. SAM semantic pseudo-GT for the 1178 train frames (stuff target for B and C).
if (-not (Test-Path $SamSemTrain)) {
    Step "SAM semantico train" @("scripts\sam_from_boxes.py", "--images", "$Train\images", "--labels", "train=$Train\labels",
        "--out", "..\sam_hist_s3_panels_train", "--save-masks", $SamSemTrain, "--no-panels", "--model", "sam_b.pt", "--device", "0")
}

# 2. SAM pseudo-GT for the validation frames (only if missing; the sweep script may have made it).
if (-not (Test-Path $GtVal)) {
    Step "SAM GT val" @("scripts\sam_from_boxes.py", "--images", "$Val\images", "--labels", "val=$Val\labels",
        "--out", "..\sam_val_panels", "--save-masks", $GtVal, "--no-panels", "--model", "sam_b.pt", "--device", "0")
}

# 3. Train B (SAM stuff masks, lr 1e-3) and C (SAM stuff masks, lr 3e-4).
Step "Entrenar B" @("scripts\train_gt_variant.py", "--data", $Data, "--out", $CkB, "--encoder", "resnet18",
    "--epochs", "150", "--seed", "0", "--device", "cuda", "--masks", $SamSemTrain, "--instances", $SamInst)
Step "Entrenar C" @("scripts\train_gt_variant.py", "--data", $Data, "--out", $CkC, "--encoder", "resnet18",
    "--epochs", "150", "--seed", "0", "--device", "cuda", "--masks", $SamSemTrain, "--instances", $SamInst, "--lr", "0.0003")

# 4. Validation evaluation (selection) for baseline A_inst2, B and C, same decoding settings.
$models = @(
    @{ name = "A_inst2"; ck = $CkBase },
    @{ name = "B_inst2_masks"; ck = "$CkB\epoch_150.pt" },
    @{ name = "C_inst2_masks_lr3e4"; ck = "$CkC\epoch_150.pt" }
)
foreach ($m in $models) {
    Step ("Val " + $m.name) @("scripts\eval_full.py", "--images", "$Val\images", "--gt", $GtVal,
        "--checkpoint", ($m.name + "=" + $m.ck), "--score-threshold", "0.05", "--top-k", "50", "--nms-iou", "0.5",
        "--device", "cuda", "--out", ("$Res\val_" + $m.name))
}

# 5. Test evaluation, once per model, reported next to the validation numbers.
foreach ($m in $models) {
    Step ("Test " + $m.name) @("scripts\eval_full.py", "--images", "$Test\images", "--gt", $GtTest,
        "--checkpoint", ($m.name + "=" + $m.ck), "--score-threshold", "0.05", "--top-k", "50", "--nms-iou", "0.5",
        "--device", "cuda", "--out", ("$Res\test_" + $m.name))
}

Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
