"""Qualitative panels for the 12 Nov meeting (2026-11-12).

1. ejemplos_fold0.jpg: test frames of scene fold 0 (unseen scenes). Columns:
   ground truth | YOLO11m | Panoptic FCN | Panoptic FCN + augmentation.
   Panoptic FCN shows stuff (river, road) as a translucent fill and each instance mask with its box.
2. ejemplos_hud.jpg: one frame of Lau's paired set, with HUD vs without HUD,
   rows YOLO11m and Panoptic FCN + augmentation.
Only frames of the exploratory scene-fold protocol (its test is already open); the S1-0 test is not used.

Run from the repo root:
    .venv\\Scripts\\python scripts\\panel_ejemplos.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT
from panoptic_mining.inference import load_image_like_training, load_model

ROOT = Path("..")
OUT = ROOT / "Claude outputs" / "ejemplos_reunion"
DATA = ROOT / "datos_escenas_v0_f0"
PAIR = ROOT / "hud_pareado_escenas_f0"
CK = {"Panoptic FCN": ROOT / "checkpoints_escenas_v0_f0/focal576_cos_rc/epoch_150.pt",
      "Panoptic FCN + aumento": ROOT / "checkpoints_escenas_v0_f0/focal576_cos_rc_aug/epoch_150.pt"}
YOLO_TEST = ROOT / "yolo_escenas_v0_f0/pred_test/labels"
NAMES = ["vehicle", "building", "road", "river", "SDZI", "machinery"]
ES = {"vehicle": "vehiculo", "building": "edificio", "machinery": "maquinaria"}
COL = {"vehicle": (235, 140, 40), "building": (40, 150, 255), "machinery": (40, 40, 230),
       "river": (230, 200, 60), "road": (200, 40, 200)}  # BGR (magenta: gray was invisible on light roads)
SHOW = 0.30  # confidence shown
W, H = 768, 432


def read_boxes(p: Path, conf: bool):
    out = []
    if p.exists():
        for l in p.read_text().splitlines():
            q = l.split()
            if len(q) >= 5:
                c = int(q[0]); xc, yc, w, h = map(float, q[1:5]); s = float(q[5]) if conf and len(q) > 5 else 1.0
                out.append((c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, s))
    return out


def title(img, text):
    cv2.rectangle(img, (0, 0), (W, 34), (25, 25, 25), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_boxes(img, boxes, min_conf=0.0, label=True):
    for c, x0, y0, x1, y1, s in boxes:
        n = NAMES[c]
        if n not in ES or s < min_conf:
            continue
        p0, p1 = (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H))
        cv2.rectangle(img, p0, p1, COL[n], 2)
        if label and s < 1.0:
            cv2.putText(img, f"{s:.2f}", (p0[0], max(12, p0[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[n], 1, cv2.LINE_AA)
    return img


def panoptic(model, path: Path, device):
    t, _ = load_image_like_training(path, getattr(model, "image_size", (512, 512)))
    with torch.no_grad():
        out = model(t.to(device))
        stuff = F.interpolate(out.stuff_logits, size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0).cpu().numpy()
        inst = []
        for name, idx in DEFAULT_CLASS_SPLIT.thing_index.items():
            kept = []
            for s, m in model.decode_instances(out, idx, top_k=50, score_threshold=SHOW):
                mk = F.interpolate(m[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0.5
                if mk.sum() < 20 or any((mk & k).sum() / (mk | k).sum() >= 0.5 for _, k in kept):
                    continue
                kept.append((float(s), mk))
            inst += [(name, s, mk) for s, mk in kept]
    return stuff, inst


def draw_panoptic(img, stuff, inst):
    over = img.copy()
    names = DEFAULT_CLASS_SPLIT.stuff_class_names
    for i, n in enumerate(names):
        if n in COL:
            over[stuff == i] = COL[n]
    for n, s, mk in inst:
        over[mk] = COL[n]
    img = cv2.addWeighted(over, 0.40, img, 0.60, 0)
    for n, s, mk in inst:
        num, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if num > 1 else 0
        x, y, w, h = st[k, :4]
        cv2.rectangle(img, (x, y), (x + w, y + h), COL[n], 2)
        cv2.putText(img, f"{s:.2f}", (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[n], 1, cv2.LINE_AA)
    return img


def load(path):
    return cv2.resize(cv2.imread(str(path)), (W, H))


def legend(width):
    bar = np.full((44, width, 3), 245, np.uint8)
    x = 10
    for n, txt in [("vehicle", "vehiculo"), ("building", "edificio"), ("machinery", "maquinaria"), ("river", "rio (relleno)"), ("road", "via (relleno)")]:
        cv2.rectangle(bar, (x, 12), (x + 24, 32), COL[n], -1)
        cv2.putText(bar, txt, (x + 32, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        x += 230
    if width > 2000:
        cv2.putText(bar, f"predicciones con confianza >= {SHOW}", (x + 20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    return bar


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {k: load_model(v, device=device) for k, v in CK.items() if v.exists()}
    man = [r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")) if r["split"] == "test"]
    seen, test = set(), []
    for r in man:
        if r["content_id"] not in seen:
            seen.add(r["content_id"]); test.append(r)

    def counts(r):
        b = read_boxes(DATA / "test/labels" / f"{r['out_name']}.txt", False)
        return [sum(1 for x in b if x[0] == c) for c in (0, 1, 5)]
    info = [(r, *counts(r)) for r in test]
    # deterministic picks: vehicles+buildings, many buildings, most machinery, river/road scene
    picks = []
    for key in (lambda x: (x[1] >= 2) * (x[1] + x[2]), lambda x: x[2], lambda x: x[3]):
        for x in sorted(info, key=key, reverse=True):
            if x[0]["out_name"] not in {p["out_name"] for p in picks}:
                picks.append(x[0]); break
    rows = []
    for r in picks:
        img_p = DATA / "test/images" / f"{r['out_name']}.jpg"
        base = load(img_p)
        gt = read_boxes(DATA / "test/labels" / f"{r['out_name']}.txt", False)
        cols = [title(draw_boxes(base.copy(), gt), f"Etiqueta (GT) - {r['nombre_frame']}"),
                title(draw_boxes(base.copy(), read_boxes(YOLO_TEST / f"{r['out_name']}.txt", True), SHOW), "YOLO11m")]
        for name, m in models.items():
            stuff, inst = panoptic(m, img_p, device)
            cols.append(title(draw_panoptic(base.copy(), stuff, inst), name))
        rows.append(np.hstack(cols))
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT / "ejemplos_fold0.jpg"), np.vstack([legend(grid.shape[1]), grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # HUD panel: one paired frame with vehicles and buildings
    pm = list(csv.DictReader(open(PAIR / "a_dataset/manifest.csv", encoding="utf-8")))
    best = max(pm, key=lambda r: (lambda b: min(sum(1 for x in b if x[0] == 0), 3) * 10 + sum(1 for x in b if x[0] == 1))(
        read_boxes(PAIR / "a_dataset/test/labels" / f"{r['out_name']}.txt", False)))
    cols = []
    for v, txt in (("b_video_con_hud", "con HUD"), ("c_video_sin_hud", "sin HUD")):
        p = PAIR / v / "test/images" / f"{best['out_name']}.png"
        base = load(p)
        yb = read_boxes(ROOT / f"resultados_hud_escenas_f0/yolo_pred_{v}/labels/{best['out_name']}.txt", True)
        col = [title(draw_boxes(base.copy(), yb, SHOW), f"YOLO11m - {txt}")]
        m = models.get("Panoptic FCN + aumento")
        if m is not None:
            stuff, inst = panoptic(m, p, device)
            col.append(title(draw_panoptic(base.copy(), stuff, inst), f"Panoptic FCN + aumento - {txt}"))
        cols.append(np.vstack(col))
    grid = np.hstack(cols)
    cv2.imwrite(str(OUT / "ejemplos_hud.jpg"), np.vstack([legend(grid.shape[1]), grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("frames:", [r["nombre_frame"] for r in picks], "| HUD:", best["nombre_frame"])
    print(f"written to {OUT}")


def busy_frames(n, min_gap=25):
    """Fold-0 test frames where something happens: YOLO detections with conf >= SHOW plus labeled
    objects, with a bonus for river/road so the stuff fill shows; frames at least min_gap apart."""
    import re
    man = [r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")) if r["split"] == "test"]
    seen, cand = set(), []
    for r in man:
        if r["content_id"] in seen:
            continue
        seen.add(r["content_id"])
        gt = read_boxes(DATA / "test/labels" / f"{r['out_name']}.txt", False)
        yb = [b for b in read_boxes(YOLO_TEST / f"{r['out_name']}.txt", True) if b[5] >= SHOW and NAMES[b[0]] in ES]
        kinds = {NAMES[b[0]] for b in gt}
        score = min(len(yb), 8) + min(sum(1 for b in gt if NAMES[b[0]] in ES), 8) + 4 * len(kinds & {"river", "road"}) \
            + 3 * len(kinds & {"vehicle", "building", "machinery"})
        cand.append((score, r))
    cand.sort(key=lambda x: -x[0])
    num = lambda r: (r["nombre_frame"].rsplit("_", 1)[0], int(re.search(r"_(\d+)$", r["nombre_frame"]).group(1)))
    picks = []
    for sc, r in cand:
        v, k = num(r)
        if all(not (num(p)[0] == v and abs(num(p)[1] - k) < min_gap) for p in picks):
            picks.append(r)
        if len(picks) == n:
            break
    return picks


def more_panels(models, device, n=6):
    picks = busy_frames(n)
    names = DEFAULT_CLASS_SPLIT.stuff_class_names
    for part in range(0, len(picks), 3):
        rows = []
        for r in picks[part:part + 3]:
            img_p = DATA / "test/images" / f"{r['out_name']}.jpg"
            base = load(img_p)
            gt = read_boxes(DATA / "test/labels" / f"{r['out_name']}.txt", False)
            cols = [title(draw_boxes(base.copy(), gt), f"Etiqueta (GT) - {r['nombre_frame']}"),
                    title(draw_boxes(base.copy(), read_boxes(YOLO_TEST / f"{r['out_name']}.txt", True), SHOW), "YOLO11m")]
            for name, m in models.items():
                stuff, inst = panoptic(m, img_p, device)
                frac = {n_: round(float((stuff == i).mean()), 3) for i, n_ in enumerate(names)}
                print(f"  {r['nombre_frame']} {name}: stuff fraction {frac}, instances {len(inst)}")
                cols.append(title(draw_panoptic(base.copy(), stuff, inst), name))
            rows.append(np.hstack(cols))
        grid = np.vstack(rows)
        out = OUT / f"ejemplos_fold0_mas_{part // 3 + 1}.jpg"
        cv2.imwrite(str(out), np.vstack([legend(grid.shape[1]), grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("written", out, [r["nombre_frame"] for r in picks[part:part + 3]])

    # two more HUD frames, YOLO-busy, away from the first one
    pm = list(csv.DictReader(open(PAIR / "a_dataset/manifest.csv", encoding="utf-8")))
    def hs(r):
        yb = [b for b in read_boxes(ROOT / f"resultados_hud_escenas_f0/yolo_pred_b_video_con_hud/labels/{r['out_name']}.txt", True) if b[5] >= SHOW]
        gt = read_boxes(PAIR / "a_dataset/test/labels" / f"{r['out_name']}.txt", False)
        return min(len(yb), 10) + 5 * min(sum(1 for b in gt if b[0] == 0), 2) + 4 * any(b[0] in (2, 3) for b in gt)
    sel = []
    for r in sorted(pm, key=hs, reverse=True):
        k = int(r["nombre_frame"].rsplit("_", 1)[1])
        if abs(k - 62) >= 25 and all(abs(k - int(x["nombre_frame"].rsplit("_", 1)[1])) >= 25 for x in sel):
            sel.append(r)
        if len(sel) == 2:
            break
    blocks = []
    for r in sel:
        cols = []
        for v, txt in (("b_video_con_hud", "con HUD"), ("c_video_sin_hud", "sin HUD")):
            p = PAIR / v / "test/images" / f"{r['out_name']}.png"
            base = load(p)
            yb = read_boxes(ROOT / f"resultados_hud_escenas_f0/yolo_pred_{v}/labels/{r['out_name']}.txt", True)
            col = [title(draw_boxes(base.copy(), yb, SHOW), f"YOLO11m - {txt} - {r['nombre_frame']}")]
            m = models.get("Panoptic FCN + aumento")
            if m is not None:
                stuff, inst = panoptic(m, p, device)
                col.append(title(draw_panoptic(base.copy(), stuff, inst), f"Panoptic FCN + aumento - {txt}"))
            cols.append(np.vstack(col))
        blocks.append(np.hstack(cols))
    grid = np.hstack(blocks) if len(blocks) == 2 else blocks[0]
    cv2.imwrite(str(OUT / "ejemplos_hud_mas.jpg"), np.vstack([legend(grid.shape[1]), grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("written ejemplos_hud_mas.jpg", [r["nombre_frame"] for r in sel])


def river_panels(models, device, n=3, min_gap=25):
    """Fold-0 test frames with the most labeled river area (frames at least min_gap apart)."""
    import re
    man = [r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")) if r["split"] == "test"]
    seen, cand = set(), []
    for r in man:
        if r["content_id"] in seen:
            continue
        seen.add(r["content_id"])
        gt = read_boxes(DATA / "test/labels" / f"{r['out_name']}.txt", False)
        area = sum((b[3] - b[1]) * (b[4] - b[2]) for b in gt if NAMES[b[0]] == "river")
        things = sum(1 for b in gt if NAMES[b[0]] in ES)
        if area > 0:
            cand.append((min(area, 0.5) + 0.02 * min(things, 5), r))
    cand.sort(key=lambda x: -x[0])
    num = lambda r: (r["nombre_frame"].rsplit("_", 1)[0], int(re.search(r"_(\d+)$", r["nombre_frame"]).group(1)))
    picks = []
    for sc, r in cand:
        v, k = num(r)
        if all(not (num(p)[0] == v and abs(num(p)[1] - k) < min_gap) for p in picks):
            picks.append(r)
        if len(picks) == n:
            break
    names = DEFAULT_CLASS_SPLIT.stuff_class_names
    rows = []
    for r in picks:
        img_p = DATA / "test/images" / f"{r['out_name']}.jpg"
        base = load(img_p)
        gt = read_boxes(DATA / "test/labels" / f"{r['out_name']}.txt", False)
        gimg = base.copy()
        for c, x0, y0, x1, y1, _ in gt:  # river/road GT boxes drawn thin as a guide
            if NAMES[c] in ("river", "road"):
                cv2.rectangle(gimg, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), COL[NAMES[c]], 2)
        cols = [title(draw_boxes(gimg, gt), f"Etiqueta (GT) - {r['nombre_frame']}"),
                title(draw_boxes(base.copy(), read_boxes(YOLO_TEST / f"{r['out_name']}.txt", True), SHOW), "YOLO11m")]
        for name, m in models.items():
            stuff, inst = panoptic(m, img_p, device)
            frac = {n_: round(float((stuff == i).mean()), 3) for i, n_ in enumerate(names)}
            print(f"  {r['nombre_frame']} {name}: stuff fraction {frac}, instances {len(inst)}")
            cols.append(title(draw_panoptic(base.copy(), stuff, inst), name))
        rows.append(np.hstack(cols))
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT / "ejemplos_rio.jpg"), np.vstack([legend(grid.shape[1]), grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("written ejemplos_rio.jpg", [r["nombre_frame"] for r in picks])


if __name__ == "__main__":
    import sys
    if "--rio" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        river_panels({k: load_model(v, device=dev) for k, v in CK.items() if v.exists()}, dev)
    elif "--mas" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        more_panels({k: load_model(v, device=dev) for k, v in CK.items() if v.exists()}, dev)
    else:
        main()
