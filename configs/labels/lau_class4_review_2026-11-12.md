# Review of class 4 ("SDZI") boxes in Lau's dataset (2026-11-12)

Scope: the 146 frames of `dataset_original` (video1 and video3, not in S1-0) that carry
class-4 boxes. 241 boxes in total. Labels in `dataset_sin_hud` are byte-identical for these frames.

Procedure: each box was drawn on its frame and labeled by Manuela in a review page
(machinery / SDZI / vehicle / other / unsure). Box index = order of class-4 lines in the label file.

Result: 232 machinery, 9 SDZI, 0 other.

| video / source | machinery | SDZI |
|---|---|---|
| video1 / dataset_split_completo | 89 | 0 |
| video1 / Adendo | 3 | 0 |
| video3 / dataset_split_completo | 125 | 0 |
| video3 / Adendo | 7 | 0 |
| video3 / yolo_dataset | 8 | 9 |

Rule: when Lau's frames are used with the 6-class scheme, a class-4 box whose decision is
`maquinaria` becomes class 5 (machinery) with the same coordinates; boxes with decision `sdzi`
stay class 4. Matching is by (frame, class4_box_index); the CSV also stores the coordinates
so the match can be verified.

Open before use in training:
- confirm with Lau/Jorge that class 4 in `dataset_split_completo` meant heavy machinery;
- check with ORB/homography that these frames share no place with the S1-0 test;
- agree with Jorge before adding them to any shared training split.
