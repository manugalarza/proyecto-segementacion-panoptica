# Presentation pack for the historical S3 seed 0 split: learning curves (train and val loss),
# qualitative predictions (A_full and A_inst2, with and without HUD on the paired test) and an
# HTML report with the metrics. Run AFTER sanity_and_full_eval.ps1 has finished:
#     powershell -ExecutionPolicy Bypass -File scripts\presentacion_hist_s3.ps1
# Output: ..\presentacion_hist_s3\  (open ..\presentacion_hist_s3\reporte\index.html)
# Log: ..\logs_presentacion_hist_s3.txt

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_presentacion_hist_s3.txt"
Start-Transcript -Path $Log -Append | Out-Null

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$TestHud = "..\test_hud_pareado_s0"
$Res = "..\resultados_hist_s3"
$CkA = "..\checkpoints_hist_s3\A_full"
$CkI = "..\checkpoints_hist_s3\A_inst2"
$Pres = "..\presentacion_hist_s3"
New-Item -ItemType Directory -Force -Path $Pres | Out-Null

# 1. Learning curves: train loss and validation loss per checkpoint (epoch 10, 20, ..., 150).
Step "Curvas de aprendizaje train y val" @("scripts\plot_training.py",
    "--run", "A_full=$CkA", "--run", "A_inst2=$CkI",
    "--val-data", $Data, "--out", "$Pres\curvas", "--device", "cuda")

# 2. Qualitative: random test frames, GT vs prediction, for each model, with and without HUD.
Step "Cualitativo A_full con HUD" @("scripts\visualize_predictions.py", "--data-root", "$TestHud\vis_con_hud",
    "--checkpoint", "$CkA\epoch_150.pt", "--split", "test", "--num-samples", "12", "--seed", "0",
    "--output-dir", "$Pres\cualitativo_A_full_con_hud")
Step "Cualitativo A_full sin HUD" @("scripts\visualize_predictions.py", "--data-root", "$TestHud\vis_sin_hud",
    "--checkpoint", "$CkA\epoch_150.pt", "--split", "test", "--num-samples", "12", "--seed", "0",
    "--output-dir", "$Pres\cualitativo_A_full_sin_hud")
Step "Cualitativo A_inst2 con HUD" @("scripts\visualize_predictions.py", "--data-root", "$TestHud\vis_con_hud",
    "--checkpoint", "$CkI\epoch_150.pt", "--split", "test", "--num-samples", "12", "--seed", "0",
    "--output-dir", "$Pres\cualitativo_A_inst2_con_hud")

# 3. HTML report: training time, loss curves, evaluation tables and the qualitative images.
Step "Reporte HTML" @("scripts\build_report.py",
    "--run", "A_full=$CkA", "--run", "A_inst2=$CkI",
    "--metrics", "$Res\completo_A_full\full_summary.json", "--metrics", "$Res\completo_A_inst2\full_summary.json",
    "--metrics", "$Res\A_full_con_hud\full_summary.json", "--metrics", "$Res\A_inst2_con_hud\full_summary.json",
    "--metrics", "$Res\A_full_sin_hud\full_summary.json", "--metrics", "$Res\A_inst2_sin_hud\full_summary.json",
    "--vis-dir", "$Pres\cualitativo_A_full_con_hud", "--out", "$Pres\reporte")

Write-Host "=== FIN  $(Get-Date)  paquete en $Pres"
Stop-Transcript | Out-Null
