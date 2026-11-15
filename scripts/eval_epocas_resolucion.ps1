# Follow-up of experimento_resolucion.ps1 (2026-11-11). Evaluation only, no training.
#   1. Checkpoint choice on VALIDATION: epochs 100-150 of F_focal_512 and F_focal_576x1024
#      (the training loss has spikes; epoch 150 fell right after one in F_focal_576x1024).
#   2. The same checkpoints on the validation frames labeled with datos_v2 only (79 of 91),
#      to see how much the 12 frames labeled with the original Etiquetas (fewer boxes per
#      frame) change the numbers.
# Test is not touched. Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\eval_epocas_resolucion.ps1
# At the end it prints one table: paste it in the chat.

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
$Log = "..\logs_eval_epocas_resolucion.txt"
Start-Transcript -Path $Log -Append | Out-Null

function Step($name, $args_) {
    Write-Host "=== $name  $(Get-Date)"
    & $Py @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "FALLO (se continua): $name exit=$LASTEXITCODE" } else { Write-Host "OK: $name" }
}

$Data = "..\datos_hist_s3_seed0"
$Ck = "..\checkpoints_resolucion"
$Res = "..\resultados_resolucion_epocas"

$ckargs = @("--checkpoint", "C_bce_512_e150=..\checkpoints_hist_s3\C_inst2_masks_lr3e4\epoch_150.pt")
foreach ($run in @("F_focal_512", "F_focal_576x1024")) {
    foreach ($e in @(100, 110, 120, 130, 140, 150)) {
        $p = "$Ck\$run\epoch_$e.pt"
        if (Test-Path $p) { $ckargs += @("--checkpoint", "$($run)_e$e=$p") }
    }
}

Step "Val completo (91)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val",
    "--device", "cuda", "--bootstrap", "500", "--out", "$Res\val_todo") + $ckargs)
Step "Val solo datos_v2 (79)" (@("scripts\eval_detection.py", "--data", $Data, "--split", "val",
    "--label-source", "datos_v2", "--device", "cuda", "--bootstrap", "500", "--out", "$Res\val_datos_v2") + $ckargs)

Write-Host "`n=== TABLA PARA PEGAR EN EL CHAT ==="
& $Py scripts\summarize_detection.py $Res | Tee-Object -FilePath "$Res\tabla.txt"
Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
