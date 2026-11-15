# EXPLORATORY (2026-11-11): the v0 scene folds (Claude outputs\folds_escenas_v0), not yet agreed
# with Jorge. Question for the thesis: how much did the frame-interleaved split (S3 seed 0, val
# 0.32 veh+bld for focal 576x1024 cosine + rel coords) overestimate performance compared with
# folds separated by scene?
# Each fold is the test set once; validation = 2 whole scene groups taken from the training side
# (65 / 86 frames). Same model as the best one so far. ~1.5 h per fold.
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\experimento_folds_escenas.ps1
foreach ($k in @(0, 1)) {
    powershell -ExecutionPolicy Bypass -File scripts\run_fixed_split.ps1 `
        -Assignment "configs\splits\escenas_v0_fold$($k)_test.csv" -Name "escenas_v0_f$k" -RelCoords -EvalTest
}
Write-Host "`n=== RESUMEN DE LOS DOS FOLDS ==="
& .venv\Scripts\python.exe scripts\summarize_detection.py ..\resultados_escenas_v0_f0
& .venv\Scripts\python.exe scripts\summarize_detection.py ..\resultados_escenas_v0_f1
