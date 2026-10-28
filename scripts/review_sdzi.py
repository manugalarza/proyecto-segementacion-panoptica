"""Review every SDZI box one by one and decide what it really is.

Shows each SDZI (class 4) box of a YOLO dataset (e.g. datos_2x2/abril) with
the full frame on the left (current box thick yellow, other SDZI boxes thin
magenta) and a zoomed crop on the right. One key per box:

    m  maquinaria   (excavator, dredge, machine)  -> becomes class 5 "machinery"
    s  SDZI         (removed soil / excavation zone, no machine is the point)
    a  ambos        (zone WITH machinery inside): stays SDZI, and you DRAW the
                     machinery boxes right away on the left panel (mouse drag;
                     z = undo last box, Enter = done). They become class 5.
    v  vehiculo     (ordinary vehicle, not machinery) -> becomes class 0
    x  error        (nothing there / wrong box) -> removed
    b  back (undo previous decision)       q  quit (progress is saved)

Decisions are written to --out (CSV) after every key, so you can stop and
resume any time. Nothing in the dataset is modified: apply the decisions
afterwards with scripts/apply_sdzi_review.py.

Usage (repo root):
    .venv\\Scripts\\python scripts\\review_sdzi.py --data "...\\datos_2x2\\abril" ^
        --out "...\\revision_sdzi.csv"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

SDZI_ID = 4
KEYS = {ord("m"): "maquinaria", ord("s"): "sdzi", ord("a"): "ambos", ord("v"): "vehiculo", ord("e"): "edificio", ord("x"): "error"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
FIELDS = ["split", "frame", "line", "xc", "yc", "w", "h", "area_pct", "decision", "machinery_boxes"]
BAND_H = 70


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_boxes(root: Path, splits: list[str]) -> list[dict]:
    boxes = []
    for split in splits:
        labels = root / split / "labels"
        if not labels.is_dir():
            continue
        for lp in sorted(labels.glob("*.txt")):
            for i, line in enumerate(lp.read_text().splitlines()):
                parts = line.split()
                if len(parts) >= 5 and int(parts[0]) == SDZI_ID:
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append({"split": split, "frame": lp.stem, "line": i, "xc": xc, "yc": yc,
                                  "w": w, "h": h, "area_pct": round(100 * w * h, 3)})
    return boxes


def render(root, box, all_in_frame, k, n, decided, panel_h=620, drawn=(), drawing=False):
    """Returns (canvas, scale) where scale maps full-image pixels to left-panel pixels."""
    img = cv2.imread(str(find_image(root / box["split"] / "images", box["frame"])))
    H, W = img.shape[:2]

    def px(b):
        return (int((b["xc"] - b["w"] / 2) * W), int((b["yc"] - b["h"] / 2) * H),
                int((b["xc"] + b["w"] / 2) * W), int((b["yc"] + b["h"] / 2) * H))

    full = img.copy()
    t = max(2, W // 640)
    for b in all_in_frame:
        cv2.rectangle(full, px(b)[:2], px(b)[2:], (255, 0, 255), t)
    x1, y1, x2, y2 = px(box)
    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 255), 3 * t)
    for (mx1, my1, mx2, my2) in drawn:
        cv2.rectangle(full, (int(mx1 * W), int(my1 * H)), (int(mx2 * W), int(my2 * H)), (0, 140, 255), 3 * t)
    full = cv2.resize(full, (int(W * panel_h / H), panel_h))

    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    m = int(0.6 * max(bw, bh)) + 40
    cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
    s = panel_h / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), panel_h), interpolation=cv2.INTER_CUBIC)
    crop = crop[:, :900]

    canvas = np.hstack([full, np.full((panel_h, 8, 3), 255, np.uint8), crop])
    band = np.zeros((70, canvas.shape[1], 3), np.uint8)
    cv2.putText(band, f"[{k + 1}/{n}] {box['split']}/{box['frame']}  box area {box['area_pct']:.2f}% of image"
                f"   decided {decided}/{n}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    help_txt = ("DIBUJA cajas de maquinaria en la imagen izquierda (arrastrar) | z borrar ultima | Enter listo"
                if drawing else
                "m maquinaria | s SDZI zona | a ambos (zona + dibujar maquina) | v vehiculo | x error | b atras | q salir")
    cv2.putText(band, help_txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 140, 255) if drawing else (0, 255, 255), 2)
    return np.vstack([band, canvas]), (full.shape[1], panel_h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="YOLO dataset root with <split>/images and <split>/labels")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    boxes = collect_boxes(args.data, args.splits)
    if not boxes:
        raise SystemExit(f"No SDZI boxes found under {args.data}")
    key_of = lambda b: (b["split"], b["frame"], int(b["line"]))  # noqa: E731
    decisions, extra = {}, {}
    if args.out.exists():
        with open(args.out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                decisions[key_of(r)] = r["decision"]
                if r.get("machinery_boxes"):
                    extra[key_of(r)] = r["machinery_boxes"]
    by_frame = {}
    for b in boxes:
        by_frame.setdefault((b["split"], b["frame"]), []).append(b)

    def save():
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for b in boxes:
                if key_of(b) in decisions:
                    w.writerow({**b, "decision": decisions[key_of(b)], "machinery_boxes": extra.get(key_of(b), "")})

    print(f"{len(boxes)} SDZI boxes in {len(by_frame)} frames; {len(decisions)} already decided.")
    k = next((i for i, b in enumerate(boxes) if key_of(b) not in decisions), len(boxes))
    history = []
    win = "Revision SDZI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mouse = {"start": None, "cur": None, "boxes": [], "panel": (1, 1)}

    def on_mouse(event, x, y, flags, param):
        pw, ph = mouse["panel"]
        fx, fy = min(max(x / pw, 0), 1), min(max((y - BAND_H) / ph, 0), 1)
        if x > pw or y < BAND_H:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["start"] = (fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["start"]:
            mouse["cur"] = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and mouse["start"]:
            (ax, ay), (bx, by) = mouse["start"], (fx, fy)
            if abs(bx - ax) > 0.005 and abs(by - ay) > 0.005:
                mouse["boxes"].append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
            mouse["start"] = mouse["cur"] = None

    cv2.setMouseCallback(win, on_mouse)

    def draw_machinery(b):
        mouse["boxes"] = []
        frame_boxes = by_frame[(b["split"], b["frame"])]
        while True:
            live = list(mouse["boxes"])
            if mouse["start"] and mouse["cur"]:
                (ax, ay), (bx, by) = mouse["start"], mouse["cur"]
                live.append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
            img, mouse["panel"] = render(args.data, b, frame_boxes, k, len(boxes), len(decisions), drawn=live, drawing=True)
            cv2.imshow(win, img)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                return "|".join(f"{(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                                for x1, y1, x2, y2 in mouse["boxes"])
            if key == ord("z") and mouse["boxes"]:
                mouse["boxes"].pop()
            if key == 27:
                return None

    while k < len(boxes):
        b = boxes[k]
        img, mouse["panel"] = render(args.data, b, by_frame[(b["split"], b["frame"])], k, len(boxes), len(decisions))
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("b") and history:
            k = history.pop()
            decisions.pop(key_of(boxes[k]), None)
            extra.pop(key_of(boxes[k]), None)
            save()
            continue
        if key == ord("a"):
            drawn = draw_machinery(b)
            if drawn is None:  # Esc cancels
                continue
            extra[key_of(b)] = drawn
        if key in KEYS:
            decisions[key_of(b)] = KEYS[key]
            save()
            history.append(k)
            k += 1
            while k < len(boxes) and key_of(boxes[k]) in decisions:
                k += 1
    cv2.destroyAllWindows()
    save()
    counts = {}
    for v in decisions.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"Saved {len(decisions)}/{len(boxes)} decisions to {args.out}: {counts}")


if __name__ == "__main__":
    main()
