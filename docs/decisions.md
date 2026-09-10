# Decision log

Open questions and decisions that affect this repo's implementation,
recorded so the reasoning survives beyond one conversation.

## RGB vs. thermal imagery (OPEN — 2026-09-09)

The dataset and the "Eyes in the Sky" system are both named around "FLIR"
/ thermal imaging, but direct inspection of the frames has only ever
shown RGB (visible-light) content. Supporting evidence found in Laura's
`proyecto-FAC` repo: its HUD-removal step detects telemetry overlays via
**HSV color masking targeting green (H 35-95) and red (H 0-15, 165-179)
hues** — a technique that only makes sense on RGB video. A raw
thermal/FLIR feed is single-channel; distinguishing "green" vs. "red" HUD
elements by hue would not be meaningful on it. The example source video in
that repo (`video30min-11to22.mp4`, 1920x1080, H.264) also carries no
thermal-camera metadata or false-color-palette indication.

This is strong but not 100% conclusive evidence (a false-color palette
applied to thermal data is technically possible, though it would make the
HSV-based HUD detection approach an odd design choice). **Not yet
confirmed** with Jorge, Fernanda, or the advisor.

**Impact if confirmed RGB**: the antecedentes section's "thermal aerial
imagery segmentation" subsection (references to Lee/thermal water,
Lee/CART, Gross/SegFly) would no longer be direct antecedents and should
be reframed; any "FLIR modality" / "thermal" language elsewhere in the
proposal document needs review. This also affects this repo: input
preprocessing should assume 3-channel RGB normalization, not single-channel
thermal, once confirmed.

**Action**: raise with the team/advisor before making further changes to
the proposal document's antecedentes section.

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
