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
