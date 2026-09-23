# Decision log

Open questions and decisions that affect this repo's implementation,
recorded so the reasoning survives beyond one conversation.

## RGB vs. thermal imagery (RESOLVED — 2026-09-10)

The dataset and the "Eyes in the Sky" system are both named around "FLIR"
/ thermal imaging, but direct inspection of the frames only ever showed
RGB (visible-light) content. Two earlier attempts at indirect evidence
(Laura's HSV-based HUD detection code, a single sample frame's tint) were
both inconclusive on their own and are kept below only as historical
context — they are not what settled this.

**Confirmed 2026-09-10: the imagery is RGB. It is never thermal.** This is
now a settled fact for this project, not an inference — treat any
remaining "thermal"/"FLIR modality" language elsewhere as needing a
one-time cleanup pass, not as an open question to re-litigate.

**Impact — now actionable**: the antecedentes section's "thermal aerial
imagery segmentation" subsection (references to Lee/thermal water,
Lee/CART, Gross/SegFly) is no longer a set of direct antecedents and
should be reframed (e.g. as adjacent/related work on segmentation in
non-standard imagery, rather than a direct comparison point) or removed
from that framing. Any "FLIR modality"/"thermal" language elsewhere in
the proposal document should be reviewed and corrected. For this repo:
input preprocessing assumes 3-channel RGB normalization — this was
already the assumption baked into `models/panoptic_fcn.py`'s
`FeatureEncoder` (`in_channels=3`), so no code change is needed here, only
documentation/proposal-text cleanup.

**Remaining action (not code)**: update the thesis proposal document's
antecedentes section and any other "FLIR"/"thermal" references to reflect
that the modality is RGB — this is Manuela's document edit, not a repo
change.

<details>
<summary>Superseded reasoning kept for context</summary>

Historical indirect evidence, kept for context even though it wasn't what
resolved this: Laura's `proyecto-FAC` HUD-removal step detects telemetry
overlays via HSV color masking targeting green (H 35-95) and red (H 0-15,
165-179) hues — a technique that only makes sense on RGB video, since a
raw thermal/FLIR feed is single-channel and distinguishing "green" vs.
"red" by hue wouldn't be meaningful on it. A single sample frame
(`frame_0000.jpg`) briefly looked bluish/monochrome on first glance, which
was suspected to be a thermal palette — but reviewing more frames showed
this was wrong (no consistent thermal look across frames; likely
atmospheric haze on that one frame). Neither of these was treated as
sufficient on its own — the resolution above came from direct
confirmation with the team/advisor instead.

</details>

## Canonical data source (RESOLVED — 2026-09-09)

Multiple similarly-named archives exist in the shared OneDrive
("Deteccion Imagenes FLIR"): `Imagenes`, `Etiquetas`, `Dataset_Balanceado*`,
`flir_base_noaug.zip`, `flir_best_division_pr.zip`, `dataset_split_completo.zip`,
`Adendo`, `video_13min_778.zip`, plus several `.pt` weight files.

Resolved by reading Jorge's `flir-leakage-pipeline` source
(`inventory.py`), which contains the explicit finding: `"Imagenes.zip +
Etiquetas.zip parecen el candidato más cercano al conjunto reportado de
1657 frames"`. His code treats `dataset_split_completo.zip`,
`Dataset_Balanceado.zip`, `Adendo.zip`, and `video_13min_778.zip` purely
as comparison/cross-check archives, not primary data sources.

**Decision**: this repo's `DATA_ROOT` (see `.env.example`) should point at
`Imagenes.zip` + `Etiquetas.zip`. `yolov11_best100.pt` is the relevant
baseline checkpoint (matches Eyes in the Sky's YOLOv11 architecture); the
other `.pt` files (`best.pt`, `bes_base_heavyaug.pt`, `yolov26_best100.pt`)
appear to belong to other experiments and are not currently used here.

## Leakage-safe split not yet available (OPEN — 2026-09-09)

Jorge's `flir-leakage-pipeline` has completed data inventory, manifest
construction, and DINOv2/CLIP feature extraction, and confirmed the
original split leaks (141 train-test overlaps). His clustering and
splitting stages — which would produce the actual leakage-safe split —
are listed in his own repo as "planned, not yet implemented."

**Decision**: proceed with the original split for the first pass of
baseline comparison (matching how Eyes in the Sky itself was evaluated),
and treat the "leakage-safe" arm of the 2×2 evaluation matrix
(`docs/methodology.md`, step 2) as blocked until Jorge's split lands.
Revisit this decision log entry once it does.

## Baseline recall comparability (RESOLVED — 2026-09-09)

Eyes in the Sky reports box-level recall (0.425 for SDZI). A panoptic
model produces masks. Comparing a raw mask-IoU recall to their box-IoU
recall would not be a fair comparison and was flagged as a required fix
in methodology review.

**Decision**: implemented `mask_based_recall_as_boxes` in
`evaluation/metrics.py`, which derives a tight bounding box from each
predicted mask and computes box-level recall against the same ground
truth and IoU threshold (0.5) the baseline used. This is the number that
gets compared to 0.425 — a mask-vs-mask recall is reported separately, if
at all, and never presented as beating the baseline.

## Thing-class count mismatch (OPEN — 2026-09-10)

The thesis objectives and antecedentes describe 6 *thing* categories
(machinery, dredges, camps, structures, containers, vehicles). The only
labels actually present in the downloaded, well-formed dataset
(`dataset_split_completo`, confirmed via `data/manifest.py` /
`panoptic-mining data manifest-summary`) are 3 thing classes — vehicle,
building, road — plus 2 stuff classes — river, SDZI. `models/panoptic_fcn.py`
and `models/context_fusion.py` default to these 3+2 counts for now (they're
just constructor defaults, easy to change), since that's what's actually
labeled and trainable today, not because the 6-category framing is
considered wrong.

**Also noted while building the manifest loader**: total frame count across
`dataset_split_completo`'s three splits is 1702, not the 1657 reported by
Eyes in the Sky (a difference of 45 frames) — not yet explained. And SDZI
is the most underrepresented class (481 boxes vs. 4120 for building), which
will need attention (class weighting or resampling) once training starts.

**Action**: resolve the 3-vs-6 category mismatch with the team/advisor —
either more classes get labeled, or the objectives/antecedentes get
reframed around what's actually annotated. Until then, don't read the
model skeleton's class-count defaults as a decision already made.

## Transfer-learning dataset: LandCover.ai (RESOLVED — 2026-09-19)

Advisor feedback: find a forest/water-body segmentation dataset for
transfer learning. Two candidates compared: LandCover.ai (aerial
orthophotos, 25-50cm/px, classes building/woodland/water/road, ~216 km²
over Poland, CC BY-NC-SA 4.0) vs. DeepGlobe Land Cover (satellite,
50cm/px, 7 classes including forest/water, 1146 tiles, custom license).

**Decision**: LandCover.ai, implemented in `data/transfer_datasets.py`.
Chosen over DeepGlobe because its imagery is aerial (same regime as the
FAC video, not satellite) and because 3 of its 4 classes map almost
directly onto what we already label (building, road as things; water as a
stuff proxy for river). DeepGlobe's extra classes (urban, agriculture,
rangeland, barren, unknown) don't correspond to anything we label. Kept as
a documented fallback in `docs/training.md`, not implemented.

**Not yet done**: the real ~1GB+ LandCover.ai archive has not been
downloaded in this session — `data/transfer_datasets.py` is implemented
and unit-tested against small synthetic image/mask pairs, not against the
real dataset. Downloading it and running `panoptic-mining train
pretrain-transfer` for real is the next step, tracked in `docs/training.md`.

**Correction (2026-09-19)**: the first version of `data/transfer_datasets.py`
assumed the download already had separate `images/`/`masks/` folders with
matching filenames. Checking the official download page and two
independent LandCover.ai loaders (torchgeo, a Hugging Face mirror) turned
up that this is wrong: the raw zip ships full-size orthophotos plus a
bundled `split.py` that must be run first, producing a tiled `output/`
directory of `{id}.jpg` + `{id}_m.png` pairs — that's the real structure
`index_landcover_ai` now reads. Caught before the real dataset was
downloaded, so no wasted download, but noted here since it changes the
setup instructions in `docs/training.md`.

## First full overnight training run (2026-09-19 to 2026-09-20)

Ran the full pretrain -> fine-tune pipeline unattended overnight per the
advisor's "debo correr todo" feedback, using `scripts/run_overnight.ps1`.

**Pretraining (LandCover.ai)**: targeted 65 epochs, reached **63/65**
before the machine slept mid-epoch (Windows suspended despite a process
running — no code error, no traceback, the log simply stops mid-step).
Checkpointing per epoch meant this cost nothing: `epoch_63.pt` was used
directly as the fine-tuning starting point instead of restarting. Fixed
going forward with `powercfg /change standby-timeout-ac 0` before the next
unattended run.

**Fine-tuning (`dataset_split_completo`, from `epoch_63.pt`)**: completed
all **213/213 epochs** cleanly. `mean_loss=0.1957` (stuff=0.1677,
thing=0.0281) — down from `mean_loss=0.6069` in the earlier 1-epoch smoke
test on the same data. No NaN (the batch-with-no-stuff-annotations fix
from `training/train.py` held up over the full run).

**What this is and isn't**: this is the first checkpoint trained for a
real number of epochs on real data, and the loss dropped substantially
compared to the 1-epoch runs. It is **not** yet evidence of a good model —
there is still no validation split or held-out metric (see
`docs/training.md`'s honesty section), so a low training loss here could
partly reflect overfitting on `dataset_split_completo`'s train split
rather than genuine generalization. The next real step is evaluating this
checkpoint (qualitative predictions, and ideally PQ/SQ/RQ against a
held-out or the independent dense-annotation subset once that exists)
before drawing any conclusion about model quality.

## Fix: explicit background class for the stuff branch (2026-09-20)

Root cause confirmed by `scripts/visualize_predictions.py` against the
overnight checkpoint: the stuff branch (river/SDZI, 2 classes) had no
"none of the above" label. Pixels outside every river/SDZI box got
`ignore_index=-1` in the loss — no gradient in either direction — so the
model never received a negative signal telling it "this is not river."
Visually this showed up exactly as predicted: ~90.6% of pixels predicted
as "river" on average across a sample, with zero relationship to where
water actually was in the frame (confirmed on a frame with an obvious
large river that the prediction didn't distinguish from the rest of the
image at all).

**Fix**: `rasterize_boxes_to_class_map` (`data/targets.py`) now accepts
an optional `background_index` — when given, unannotated pixels get that
explicit class instead of -1. `ClassSplit` (`data/torch_dataset.py`) gets
a `stuff_background_class_name` field; `DEFAULT_CLASS_SPLIT` now has 3
stuff classes (`river`, `SDZI`, `none`) instead of 2, with `"none"` as
the background class. This only changes `data/torch_dataset.py` (our own
`dataset_split_completo` data) — `data/transfer_datasets.py` (LandCover.ai
pretraining) keeps the old ignore-index behavior for now, tracked as a
follow-up, since only the pretrained *encoder* gets reused for
fine-tuning (the pretrain heads are discarded either way, so this
asymmetry doesn't block using the existing `epoch_63.pt` pretrain
checkpoint with the fixed fine-tuning data).

**Not yet re-run**: this is a data/model shape change (stuff head is now
3-way, not 2-way), so the existing `checkpoints/finetune_overnight/`
checkpoints are stale — they were trained against the old 2-class stuff
target and won't load into the new 3-class head. Re-running `train run`
from the same `epoch_63.pt` pretrain checkpoint (only the encoder loads
from it, so this is unaffected) is the next step, not done yet as of this
entry.

The thing branch's low-confidence issue (see the same visualization run:
vehicle went from 0 detections at score_threshold=0.3 to 9 at 0.05,
building/road similarly jumped) looks different — not a design gap like
the missing background class, more likely just needing more
epochs/better-calibrated thresholds. Not changed in this fix; tracked as
a possible follow-up (per-class loss weighting) if it doesn't improve
with a longer run.

## Fix: LandCover.ai thing-instance boxes computed on the downsampled mask (2026-09-20)

Surfaced by your first real `pytest` run with torch installed since
`data/transfer_datasets.py` was written (`test_landcover_dataset_thing_heatmap_has_building_peak`
failed: `thing_heatmap.max() == 0.0`). Root cause: `LandCoverAIDataset.__getitem__`
ran connected-component detection on `mask_resized` — the mask *after*
downsampling to the coarse target grid (e.g. 64x64 for a 512x512 tile at
stride 8) — so a small-but-real building blob could shrink to 0-1 pixels
via nearest-neighbor resizing and then get silently dropped by the
noise-area filter (`area < 4`) in `_connected_component_boxes`. Not a
data-collapse issue like the stuff-branch one above — a straightforward
loss of real instances during downsampling.

**Fix**: detect connected components on the native-resolution mask
instead, then scale the resulting box coordinates down to target
resolution (same pattern `data/torch_dataset.py` already uses for real
YOLO boxes). See the regression test's docstring in
`tests/test_transfer_datasets.py` for the exact mechanics.

**Impact**: this only affects `train pretrain-transfer` (LandCover.ai);
it does not touch the already-completed overnight pretraining checkpoint
(`epoch_63.pt`) or require redoing it — pretraining would need to be
re-run only if the LandCover.ai thing-branch pretraining signal itself
matters going forward (it's the *encoder* that gets reused for
fine-tuning, and encoder weights are unaffected by this bug since it's
about the pretraining thing-head's targets specifically, not the shared
encoder's stuff-branch or image inputs).

## Stuff-branch background-class fix: confirmed working (2026-09-20/21)

Re-ran fine-tuning from `epoch_63.pt` with the 3-class stuff target
(river/SDZI/none) — `checkpoints/finetune_fixed/epoch_213.pt`,
mean_loss=0.2225 (stuff=0.1953, thing=0.0272; not directly comparable to
the pre-fix run's stuff=0.1677, since the pre-fix loss only ever counted
annotated pixels while this one counts every pixel — a fair count is
harder, not necessarily worse).

**Confirmed both quantitatively and visually.** Sample stats across 8 val
frames: river mean pixel fraction dropped from 0.906 (pre-fix, collapsed)
to **0.037**; none (background) is now 0.963 — a plausible distribution
given water is genuinely a small fraction of most frames. Visually,
`scripts/visualize_predictions.py` on a frame with an obvious real river:
the prediction now colors specifically the diagonal water region, not
the whole frame — and is actually *more* spatially precise than its own
coarse box-rasterized training target (whose GT box for this frame
covers both the compound and the water; the prediction correctly
excludes the compound). The stuff branch has real learned spatial
structure now, not a global class bias.

**Thing branch remains a separate, unresolved issue** — if anything
slightly weaker after this re-run at score_threshold=0.3 (vehicle: 0,
road: 0, building: 1/frame, vs. vehicle: 0, road: 3/8, building: 9/8
before this fix). Most likely explanation: shared-encoder training
dynamics shifted slightly with a harder stuff task, not a regression
caused by the fix itself. Still consistent with the earlier finding
(low-threshold check showed real but weak signal, not total collapse) —
tracked as a follow-up (per-class loss weighting, more epochs, and/or a
lower default score_threshold), not addressed in this fix.

**Low-threshold check confirms it's calibration, not collapse**: at
score_threshold=0.05 (vs. the default 0.3), the same checkpoint produces
vehicle: 0->13, building: 1->66, road: 0->65 detections across the same
8 frames. Real, weak peaks exist for all three thing classes; they just
don't clear the default 0.3 threshold yet. Same pattern observed before
this fix (see the entry above), so the stuff-branch change didn't affect
this — it's an independent, still-open issue.

## Thing-branch class weighting (2026-09-21, part of the Thursday plan)

Added `training/train.py`'s `compute_losses` an optional `thing_pos_weight`
(a per-class weight for `binary_cross_entropy_with_logits`), and
`data/torch_dataset.compute_thing_pos_weight` to derive it from the
manifest's real class counts by inverse frequency (building 4120 -> 1.0,
vehicle 717 -> ~5.75, road 1958 -> ~2.10). Wired into `panoptic-mining
train run` as the default (`--no-weight-thing-classes` to disable).
Motivation: the 2026-09-20 visualizations showed vehicle/road detections
had real but much weaker signal than building even before the stuff-branch
fix — this doesn't address the separate low-confidence-across-the-board
issue directly, but should stop the loss from being dominated by
whichever class has the most boxes. Untested against real data as of
this entry — same caveat as everything torch-dependent in this project;
the next real fine-tuning run against `dataset_split_completo` is what
will actually tell us if it helps.
