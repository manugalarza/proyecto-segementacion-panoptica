# HUD ablation on the scene folds (exploratory protocol; its test folds are already open), 2026-11-12.
# For every test frame of each fold that Lau's paired dataset covers (video_13min only), three versions
# of the same frame with the SAME labels:
#   a_dataset        image of the dataset (what the models were trained on)
#   b_video_con_hud  Lau's frame extracted from the video, HUD kept
#   c_video_sin_hud  Lau's frame extracted from the video, HUD removed
# b vs c = effect of the HUD alone; a vs b = effect of the image source/quality.
# Runs on CPU so it does not compete with the GPU training. Does NOT touch the S1-0 test.
#
# Run from the repo root (any time, in a new window):
#     powershell -ExecutionPolicy Bypass -File scripts\ablacion_hud_escenas.ps1
# At the end it prints one table per fold: paste it in the chat.

$ErrorActionPreference = "Continue"
$Py = ".venv\Scripts\python.exe"
Start-Transcript -Path "..\logs_ablacion_hud_escenas.txt" -Append | Out-Null

# Wait until fold 0 of experimento_folds_escenas_2.ps1 is finished (its YOLO and the augmented model),
# so no half-trained checkpoint is used.
$FoldsLog = "..\logs_experimento_folds_escenas_2.txt"
while (-not (Select-String -Path $FoldsLog -Pattern "=== YOLO fold 1|=== FIN" -Quiet)) {
    Write-Host "Esperando a que termine el fold 0  $(Get-Date)"; Start-Sleep -Seconds 120
}

foreach ($k in @(0, 1)) {
    $Pair = "..\hud_pareado_escenas_f$k"
    $Res = "..\resultados_hud_escenas_f$k"
    $Ck = @("--checkpoint", "ours_e150=..\checkpoints_escenas_v0_f$k\focal576_cos_rc\epoch_150.pt")
    $Aug = "..\checkpoints_escenas_v0_f$k\focal576_cos_rc_aug\epoch_150.pt"
    if (Test-Path $Aug) { $Ck += @("--checkpoint", "ours_aug_e150=$Aug") }
    $Yolo = "..\yolo_escenas_v0_f$k\train\weights\best.pt"
    foreach ($v in @("a_dataset", "b_video_con_hud", "c_video_sin_hud")) {
        Write-Host "=== fold $k $v  $(Get-Date)"
        $Pred = @()
        $YoloDone = $false
    if (Test-Path "..\yolo_escenas_v0_f$k\train\results.csv") { $YoloDone = ((Get-Content "..\yolo_escenas_v0_f$k\train\results.csv").Count -ge 61) }
    if ($YoloDone -and (Test-Path $Yolo)) {
            & $Py scripts\yolo_predict_dir.py --weights $Yolo --images "$Pair\$v\test\images" --out "$Res\yolo_pred_$v" --device cpu
            $Pred = @("--predictions", "YOLO11m=$Res\yolo_pred_$v\labels")
        }
        & $Py scripts\eval_detection.py --data "$Pair\$v" --split test --device cpu --bootstrap 1000 --out "$Res\$v" @Ck @Pred
    }
    Write-Host "`n--- fold $k ($((Get-ChildItem "$Pair\a_dataset\test\images").Count) frames pareados) ---"
    & $Py scripts\summarize_detection.py $Res | Tee-Object -FilePath "$Res\tabla.txt"
}
Write-Host "=== FIN  $(Get-Date)"
Stop-Transcript | Out-Null
