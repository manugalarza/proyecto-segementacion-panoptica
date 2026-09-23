# Chains pretrain-transfer -> fine-tune for an unattended overnight run,
# with full output logged to timestamped files (so you don't need to keep
# watching the terminal, and can check progress/errors in the morning).
#
# Usage (from the repo root, inside your activated venv):
#   powershell -ExecutionPolicy Bypass -File scripts\run_overnight.ps1
#
# Before running, edit the two paths below to match your machine.

$LandcoverOutput = "C:\Users\manue\OneDrive\Documentos\Universidad\2026-2\Proyecto de grado\landcover.ai.v1\output"
$LandcoverSplit  = "C:\Users\manue\OneDrive\Documentos\Universidad\2026-2\Proyecto de grado\landcover.ai.v1\train.txt"
$DataRoot        = "..\imagenes recortado\dataset_split_completo\dataset_split_completo"

# Epoch counts below are calibrated from REAL timed runs on this machine
# (2026-09-19, Measure-Command over 2 epochs each):
#   - pretrain-transfer: 548.1837953s / 2 = 274.09s/epoch (~4.57 min/epoch)
#   - fine-tune (train run): 101.2614884s / 2 = 50.63s/epoch (~0.84 min/epoch)
# Budget: 8 hours total, split 5h pretrain / 3h fine-tune (more time on
# pretrain since the fine-tune stage reuses whatever encoder it produces).
#   65 pretrain epochs  ~= 65 * 274.09s / 60 = 296.9 min (~4.95 h)
#   213 fine-tune epochs ~= 213 * 50.63s / 60 = 179.7 min (~3.00 h)
#   total ~= 476.6 min (~7.94 h) — leaves a small buffer under 8h for
#   checkpoint I/O and other overhead not captured in the 2-epoch timing.
# If you change how many hours you're leaving this running, recompute:
#   epochs = (hours_for_that_stage * 3600) / seconds_per_epoch_above
$PretrainEpochs = 65
$FinetuneEpochs = 213

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$PretrainLog = "$LogDir\pretrain_$Timestamp.log"
$FinetuneLog = "$LogDir\finetune_$Timestamp.log"

Write-Host "Starting pretrain-transfer ($PretrainEpochs epochs) -- logging to $PretrainLog"
python -m panoptic_mining.cli train pretrain-transfer $LandcoverOutput `
    --split-file $LandcoverSplit `
    --epochs $PretrainEpochs `
    --checkpoint-dir checkpoints\pretrain_overnight `
    *>&1 | Tee-Object -FilePath $PretrainLog

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pretrain step failed (exit code $LASTEXITCODE) -- see $PretrainLog. Skipping fine-tune."
    exit 1
}

$PretrainCheckpoint = "checkpoints\pretrain_overnight\epoch_$PretrainEpochs.pt"
Write-Host "Starting fine-tune ($FinetuneEpochs epochs) from $PretrainCheckpoint -- logging to $FinetuneLog"
python -m panoptic_mining.cli train run $DataRoot `
    --pretrained-checkpoint $PretrainCheckpoint `
    --epochs $FinetuneEpochs `
    --checkpoint-dir checkpoints\finetune_overnight `
    *>&1 | Tee-Object -FilePath $FinetuneLog

Write-Host "Done. Check $PretrainLog and $FinetuneLog for the full loss history."
    