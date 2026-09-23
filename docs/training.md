# Training and transfer learning

Advisor feedback (2026-09-19): (1) actually run the pipeline end to end,
not just have the pieces implemented in isolation; (2) find a forest/water
body segmentation dataset to use for transfer learning, since river and
SDZI — our two stuff classes — are exactly where the small, imbalanced
mining dataset hurts most (SDZI: 481 boxes vs. 4120 for building, see
`decisions.md`).

## Dataset chosen for transfer learning: LandCover.ai

Two candidates were evaluated:

| | [LandCover.ai](https://landcover.ai.linuxpolska.com/) (Boguszewski et al., CVPR-W 2021) | [DeepGlobe Land Cover](https://arxiv.org/abs/1805.06561) (2018) |
|---|---|---|
| Imagery | Aerial orthophotos, 25-50 cm/px | Satellite, 50 cm/px |
| Classes | building, woodland, water, road | urban, agriculture, rangeland, forest, water, barren, unknown |
| Size | 41 orthophotos, ~216 km² (Poland) | 1,146 tiles (803 train / 171 val / 172 test), 2448x2448 px each |
| License | CC BY-NC-SA 4.0 | Custom (DeepGlobe rules + DigitalGlobe internal-use + annotation license) |
| Download | [landcover.ai.v1.zip](https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip) | [Kaggle mirror](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) |

**Chosen: LandCover.ai.** Two reasons:

1. **Domain match**: it's aerial orthophoto imagery, the same imaging
   regime as the FAC video (a drone/aircraft looking down at close range),
   not satellite. DeepGlobe's imagery is from a different altitude/sensor
   and the visual texture is noticeably different.
2. **Class match**: its 4 classes (building, woodland, water, road) line
   up almost one-to-one with what's actually labeled in our own dataset
   (building/road as things, water ~ river as stuff) — see
   `data/transfer_datasets.py` for the exact mapping. DeepGlobe's `forest`
   and `water` classes are also usable, but its other 5 classes (urban,
   agriculture, rangeland, barren, unknown) don't correspond to anything
   we label, which is more noise for less direct benefit than LandCover.ai
   gives here.

DeepGlobe is kept as a documented fallback (e.g. if LandCover.ai's license
turns out to be a problem, or if more data volume matters more than domain
match once a first pretraining run is evaluated) but is not implemented in
`data/transfer_datasets.py` — only LandCover.ai is.

**Not yet done**: the ~1GB+ LandCover.ai zip was not downloaded in this
session (no network path to fetch and stage it against the real dataset
here) — `data/transfer_datasets.py` and its tests use small synthetic
image/mask pairs instead. Downloading the real archive and pointing
`panoptic-mining train pretrain-transfer` at it is the next real step, not
done yet.

### How to download and prepare it (do this on your own machine)

1. Download the zip: [landcover.ai.v1.zip](https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip)
   (check the file size before waiting on a slow connection — it's roughly
   1-2 GB).
2. Extract it. The extracted folder contains full-size orthophotos and
   masks, plus a bundled `split.py` script and `train.txt`/`val.txt`/`test.txt`.
   **Don't point the CLI at this top-level folder directly** — the images
   here are full-size orthophotos (thousands of pixels per side), not the
   512x512 tiles the loader expects.
3. Run `split.py` from inside the extracted folder to tile everything into
   512x512 pieces. The exact command-line flags weren't verified against
   the real file in this session — open `split.py` in a text editor (or
   run `py split.py --help`) to see what it actually expects before
   running it; it's a short, readable script. It should produce an
   `output/` directory of `{id}.jpg` + `{id}_m.png` pairs.
4. Point the CLI at that `output/` directory:
   ```bash
   panoptic-mining train pretrain-transfer path/to/extracted/output \
       --split-file path/to/extracted/train.txt --epochs 5
   ```
   Omit `--split-file` to train on every tile under `output/` regardless
   of the official train/val/test assignment (fine for a first pretraining
   run; use it once you want a proper held-out check on the pretraining
   task itself).

If step 3 turns out to need a dependency this project doesn't already have
(e.g. a specific `rasterio`/`gdal` version for reading the raw `.tif`
orthophotos), that's expected — LandCover.ai's own tooling has its own
requirements, separate from this repo's.

## Workflow: pretrain, then fine-tune

```bash
# 1. Pretrain the shared encoder on LandCover.ai (forest/water/building/road).
panoptic-mining train pretrain-transfer path/to/landcover.ai.v1 \
    --epochs 5 --checkpoint-dir checkpoints/pretrain

# 2. Fine-tune on our own mining dataset, starting from that encoder.
panoptic-mining train run path/to/dataset_split_completo \
    --pretrained-checkpoint checkpoints/pretrain/epoch_5.pt \
    --epochs 20 --checkpoint-dir checkpoints/finetune
```

Step 2 loads only the *encoder* weights from step 1's checkpoint — the
stuff/thing heads are re-initialized at our own class counts (2 stuff, 3
thing), since LandCover.ai's classes (2 stuff: woodland/water; 2 thing:
building/road) don't line up index-for-index with ours. This is the
standard "pretrain backbone, replace head" transfer-learning pattern, and
it falls out of `PanopticFCN.__init__` already accepting an `encoder`
argument — no change to `models/panoptic_fcn.py` was needed to support it.

## What "running everything end to end" means today, honestly

`panoptic-mining train run` and `train pretrain-transfer` are runnable and
have now both been confirmed against real data end to end. First smoke
test (2026-09-19, 1 epoch each): `train pretrain-transfer` on the real
LandCover.ai download (mean_loss=0.4896); `train run` (fine-tuning) on
`dataset_split_completo`, starting from that pretrained checkpoint
(mean_loss=0.6069, stuff=0.5505, thing=0.0564, no NaN) after the fix
below.

**First full overnight run (2026-09-19 -> 2026-09-20)**, via
`scripts/run_overnight.ps1`: pretraining reached 63/65 targeted epochs
(the machine slept mid-run — see `docs/decisions.md`; the per-epoch
checkpoint meant `epoch_63.pt` was simply used as-is). Fine-tuning from
that checkpoint then ran the full **213/213 epochs**, finishing at
**mean_loss=0.1957 (stuff=0.1677, thing=0.0281)** — a real drop from the
1-epoch smoke test's 0.6069 on the same data. **The full pretrain ->
fine-tune chain now runs unattended, overnight, on real data, for real,**
which is what the advisor's "run everything" feedback asked for. This is
still not a claim the resulting checkpoint is *good*: there is no
validation split yet (see gap #3 below), so a training loss this low can
also mean overfitting on `dataset_split_completo`'s train split rather
than genuine generalization — the honest next step is looking at what
this checkpoint actually predicts, not just its training loss. Two real
bugs were caught getting here, both fixed:

- **NaN loss when a batch has zero stuff-annotated pixels.**
  `nn.functional.cross_entropy(..., ignore_index=-1)` divides by the count
  of non-ignored pixels; most frames in `dataset_split_completo` have
  vehicle/building/road boxes but no river/SDZI box (stuff classes are
  comparatively rare — see the class-imbalance note in `decisions.md`), so
  a batch can easily have an all -1 stuff target. That count-of-zero case
  returns NaN, which then poisons every downstream average — observed as
  `epoch 1/1: mean_loss=nan` running `train run` for real. Fixed in
  `compute_losses` (`training/train.py`): a batch with no annotated stuff
  pixels now contributes an exact, finite zero to the stuff loss instead.
  See `tests/test_train.py`'s two regression tests for this.
- **3-channel LandCover.ai mask files** — see `data/transfer_datasets.py`'s
  module docstring; already covered above.

Three more things are explicitly not covered yet, tracked here rather
than glossed over:

1. **No per-instance mask supervision.** The kernel-generator and
   mask-feature heads (`PanopticFCN.kernel_head`, `.mask_feature_head`)
   run in the forward pass but aren't supervised by any loss in
   `training/train.py` yet — only the stuff and thing-heatmap heads are.
   Adding this needs per-instance mask targets, which is where
   `data/points.py`'s pseudo-masks come back in (as training targets, not
   just an evaluation-time convenience) — not implemented yet.
2. **Coarse stuff targets.** `data/torch_dataset.py` rasterizes a box's
   *entire* rectangle as its class for the stuff-branch target, not the
   refined GrabCut/flood-fill pseudo-mask from `data/points.py` — running
   pseudo-mask generation per-sample inside the data loader was judged too
   slow; a precomputed pseudo-mask cache is the natural fix, not built.
   Separately (found via `scripts/visualize_predictions.py` against the
   overnight checkpoint, fixed 2026-09-20, see `docs/decisions.md`): the
   stuff branch also had no explicit background class, so it never
   learned any real spatial structure at all — now fixed with a 3rd
   "none" stuff class, but the fine-tuning checkpoints need a re-run
   against the new 3-class target before this gap is actually closed in
   a trained checkpoint, not just in the code.
3. **No validation loop / early stopping / hyperparameter search** — just
   a training loop over one dataset split for a fixed number of epochs.

None of this blocks getting a first, honest, if rough, trained checkpoint
and starting to look at what it produces — which is the actual next step.
