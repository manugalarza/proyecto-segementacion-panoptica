# Queue for the night before the 8 am meeting (2026-11-11).
#   1. Waits until experimento_folds_escenas_2.ps1 finishes FOLD 0 (YOLO + our model with augmentation),
#      i.e. until its log shows "=== YOLO fold 1". Then it stops that run (process tree of PID 23664).
#   2. Panoptic FCN on Jorge's historical-S1-0, scenario A (datos_v2 + machinery). Validation only.
#   3. Same, scenario B (original FAC labels). Validation only.
#   4. Resumes experimento_folds_escenas_2.ps1 for fold 1 (fold 0 steps are skipped or only re-evaluated).
#      This one will NOT be done by 8 am; it just uses the GPU afterwards.
# The test set of S1-0 is never evaluated here.
#
# Run from the repo root in a NEW PowerShell window (leave the current one running):
#     powershell -ExecutionPolicy Bypass -File scripts\cola_despues_de_folds.ps1
# Keep the laptop plugged in.

param([int]$WaitPid = 23664)

$Log = "..\logs_experimento_folds_escenas_2.txt"
$Assign = "..\FLIR_splits_para_revision\manuela-historical-v2\historical-S1-0.csv"
try { powercfg /change standby-timeout-ac 0 } catch { }

function Fold0Done {
    if (-not (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue)) { return $true }
    if (Test-Path $Log) {
        if (Select-String -Path $Log -Pattern "=== YOLO fold 1" -Quiet) { return $true }
        if (Select-String -Path $Log -Pattern "=== FIN" -Quiet) { return $true }
    }
    return $false
}

Write-Host "Esperando a que termine el fold 0 (PID $WaitPid)  $(Get-Date)"
while (-not (Fold0Done)) { Start-Sleep -Seconds 120; Write-Host "  fold 0 sigue  $(Get-Date)" }

if (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Write-Host "Fold 0 listo. Deteniendo el experimento de folds (fold 1 se reanuda al final)  $(Get-Date)"
    taskkill /PID $WaitPid /T /F | Out-Null
}
Start-Sleep -Seconds 60   # let the GPU free up

Write-Host "`n=== RESULTADOS FOLD 0 (ronda 2) ==="
& .venv\Scripts\python.exe scripts\summarize_detection.py "..\resultados_escenas_v0_f0_ronda2" | Tee-Object -FilePath "..\resultados_escenas_v0_f0_ronda2\tabla.txt"

Write-Host "=== S1-0 escenario A  $(Get-Date)"
& powershell -ExecutionPolicy Bypass -File scripts\run_fixed_split.ps1 -Assignment $Assign -Name hist_s1_0 -RelCoords
Write-Host "=== S1-0 escenario B  $(Get-Date)"
& powershell -ExecutionPolicy Bypass -File scripts\run_fixed_split.ps1 -Assignment $Assign -Name hist_s1_0_orig -RelCoords -OriginalLabels

Write-Host "`n=== TABLAS PARA PEGAR EN EL CHAT ==="
Write-Host "--- Folds por escena, fold 0 como test (YOLO vs nuestro modelo con/sin aumento) ---"
Get-Content "..\resultados_escenas_v0_f0_ronda2\tabla.txt" -ErrorAction SilentlyContinue
Write-Host "--- S1-0 escenario A (datos_v2), solo validacion ---"
Get-Content "..\resultados_hist_s1_0\tabla.txt" -ErrorAction SilentlyContinue
Write-Host "--- S1-0 escenario B (etiquetas originales), solo validacion ---"
Get-Content "..\resultados_hist_s1_0_orig\tabla.txt" -ErrorAction SilentlyContinue
Write-Host "=== S1-0 TERMINADO  $(Get-Date)"

# The interrupted fold-1 YOLO run must start from scratch (otherwise a partial last.pt would be reused).
$Y1 = "..\yolo_escenas_v0_f1"
if (Test-Path $Y1) {
    $rows = 0
    if (Test-Path "$Y1\train\results.csv") { $rows = (Get-Content "$Y1\train\results.csv").Count - 1 }
    if ($rows -lt 60) { Write-Host "Borrando YOLO fold 1 incompleto ($rows epocas)"; Remove-Item -Recurse -Force $Y1 }
}
Write-Host "`n=== Reanudando folds (fold 1)  $(Get-Date)"
& powershell -ExecutionPolicy Bypass -File scripts\experimento_folds_escenas_2.ps1
