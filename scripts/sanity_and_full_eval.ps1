# Sanity check on a train subset (every 8th frame) + full evaluation on the 271-frame HUD test
# of the historical S3 seed 0 split (A_full and A_inst2). Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\sanity_and_full_eval.ps1
# Log: ..\logs_sanity_full_eval.txt

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_sanity_full_eval.txt"
Start-Transcript -Path $Log -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$Res = "..\resultados_hist_s3"
$CkA = "..\checkpoints_hist_s3\A_full\epoch_150.pt"
$CkI = "..\checkpoints_hist_s3\A_inst2\epoch_150.pt"

# 0. Build the train subset (every 8th frame, sorted by name) with images and labels.
$San = "..\sanity_train_s0"
New-Item -ItemType Directory -Force -Path "$San\images", "$San\labels" | Out-Null
$files = Get-ChildItem "$Data\train\images" | Sort-Object Name
$i = 0; $n = 0
foreach ($f in $files) {
    if ($i % 8 -eq 0) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
        Copy-Item $f.FullName "$San\images\" -Force
        Copy-Item "$Data\train\labels\$stem.txt" "$San\labels\" -Force
        $n++
    }
    $i++
}
Write-Host "Subset de train: $n frames"

# 1. SAM pseudo-GT for the train subset and for the full 271-frame test.
Step "SAM GT train subset" @("scripts\sam_from_boxes.py", "--images", "$San\images", "--labels", "train=$San\labels",
    "--out", "..\sanity_sam_panels", "--save-masks", "$San\gt", "--no-panels", "--model", "sam_b.pt", "--device", "0")
Step "SAM GT test completo" @("scripts\sam_from_boxes.py", "--images", "$Data\test\images", "--labels", "test=$Data\test\labels",
    "--out", "..\sam_test_full_panels", "--save-masks", "..\gt_test_full_hist_s0", "--no-panels", "--model", "sam_b.pt", "--device", "0")

# 2. Sanity: A_full on frames it was trained on (should score well if pipeline is correct).
Step "Sanidad A_full en train" @("scripts\eval_full.py", "--images", "$San\images", "--gt", "$San\gt",
    "--checkpoint", "hist_A_full=$CkA", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\sanidad_train_A_full")

# 3. Full evaluation on the 271-frame HUD test (things, machinery and stuff).
Step "Eval completo A_full test HUD" @("scripts\eval_full.py", "--images", "$Data\test\images", "--gt", "..\gt_test_full_hist_s0",
    "--checkpoint", "hist_A_full=$CkA", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\completo_A_full")
Step "Eval completo A_inst2 test HUD" @("scripts\eval_full.py", "--images", "$Data\test\images", "--gt", "..\gt_test_full_hist_s0",
    "--checkpoint", "hist_A_inst2=$CkI", "--nms-iou", "0.5", "--device", "cuda", "--out", "$Res\completo_A_inst2")

Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
