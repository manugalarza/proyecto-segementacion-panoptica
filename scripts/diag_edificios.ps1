# Diagnostic of the building (and vehicle) misses on VALIDATION, no training (2026-11-11).
# For each checkpoint it reports, besides AP: recall at IoU 0.5 vs 0.3 (found but badly
# localized), recall by box size, and "merged" detections (one confident box covering two or
# more labeled buildings). Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\diag_edificios.ps1
# Paste the "Diagnostics" block printed at the end.

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
Start-Transcript -Path "..\logs_diag_edificios.txt" -Append | Out-Null
$Ck = "..\checkpoints_resolucion"
& $Py scripts\eval_detection.py --data ..\datos_hist_s3_seed0 --split val --device cuda --bootstrap 200 `
    --checkpoint "C_bce_512_e150=..\checkpoints_hist_s3\C_inst2_masks_lr3e4\epoch_150.pt" `
    --checkpoint "F_focal_512_e110=$Ck\F_focal_512\epoch_110.pt" `
    --checkpoint "F_focal_512_e140=$Ck\F_focal_512\epoch_140.pt" `
    --checkpoint "F_focal_576x1024_e100=$Ck\F_focal_576x1024\epoch_100.pt" `
    --checkpoint "F_focal_576x1024_e130=$Ck\F_focal_576x1024\epoch_130.pt" `
    --checkpoint "F_focal_576x1024_e150=$Ck\F_focal_576x1024\epoch_150.pt" `
    --out ..\resultados_diag_edificios
Stop-Transcript | Out-Null
