# Tune the instance decoding of A_inst2 on the VALIDATION split (not on test).
# Builds SAM pseudo-GT for the 107 val frames, then evaluates a small grid of
# score threshold / top-k / NMS settings. Test is not touched.
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\sweep_val_A_inst2.ps1
# Log: ..\logs_sweep_val_A_inst2.txt

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_sweep_val_A_inst2.txt"
Start-Transcript -Path $Log -Append | Out-Null
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "No se pudo cambiar la suspension." }

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Val = "..\datos_hist_s3_seed0\val"
$CkI = "..\checkpoints_hist_s3\A_inst2\epoch_150.pt"
$Gt = "..\gt_val_hist_s0"
$Res = "..\resultados_hist_s3\val_sweep"

# 1. SAM pseudo-GT for the validation frames (same convention as the test GT).
Step "SAM GT val" @("scripts\sam_from_boxes.py", "--images", "$Val\images", "--labels", "val=$Val\labels",
    "--out", "..\sam_val_panels", "--save-masks", $Gt, "--no-panels", "--model", "sam_b.pt", "--device", "0")

# 2. Grid over decoding settings.
foreach ($sc in @("0.05", "0.02", "0.01")) {
    foreach ($tk in @("50", "150")) {
        foreach ($nm in @("0.5", "0.3")) {
            $name = "s" + $sc.Replace(".", "") + "_k" + $tk + "_n" + $nm.Replace(".", "")
            Step "Val $name" @("scripts\eval_full.py", "--images", "$Val\images", "--gt", $Gt,
                "--checkpoint", "hist_A_inst2=$CkI", "--score-threshold", $sc, "--top-k", $tk,
                "--nms-iou", $nm, "--device", "cuda", "--out", "$Res\$name")
        }
    }
}

Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
