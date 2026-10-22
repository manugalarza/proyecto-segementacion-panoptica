"""Build a self-contained HTML page to review SDZI boxes in the browser and
export the decisions as JSON (same fields as review_sdzi.py's CSV).

By default it lists the SDZI (class 4) boxes of --source that were NOT already
reviewed, i.e. frames that are not in --reviewed (e.g. the scene frames
frame_*_scene* and the video gap frames left out of datos_2x2).

The page works offline (images embedded as base64), saves progress in the
browser, and exports revision_sdzi_<n>.json. It contains FAC images: keep the
file local, do not upload or share it.

Keys in the page:  m maquinaria | s SDZI zona | a ambos (then drag on the left
image to draw the machinery boxes, z undo, Enter done) | v vehiculo | x error |
e edificio (structure, e.g. cambuche) | b / left arrow back | right arrow skip.

Usage (server or Lenovo, repo root):
    .venv\\Scripts\\python scripts\\make_review_html.py ^
        --source "C:\\...\\OneDrive_1_7-10-2026\\dataset_split_completo" ^
        --reviewed "C:\\...\\datos_2x2\\abril" --out "C:\\...\\revision_sdzi_escenas.html"
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2

SDZI_ID = 4
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def b64jpg(img, quality=82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="YOLO dataset root (<split>/images, <split>/labels)")
    ap.add_argument("--reviewed", type=Path, default=None, help="Dataset root whose frames were already reviewed (skipped)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--full-width", type=int, default=960)
    args = ap.parse_args()

    done = set()
    if args.reviewed:
        for lp in args.reviewed.rglob("labels/*.txt"):
            done.add(lp.stem)

    items, missing = [], []
    for split in args.splits:
        lab_dir, img_dir = args.source / split / "labels", args.source / split / "images"
        if not lab_dir.is_dir():
            continue
        for lp in sorted(lab_dir.glob("*.txt")):
            if lp.stem in done:
                continue
            lines = lp.read_text().splitlines()
            sdzi = [(i, l.split()) for i, l in enumerate(lines) if len(l.split()) >= 5 and int(l.split()[0]) == SDZI_ID]
            if not sdzi:
                continue
            ip = find_image(img_dir, lp.stem)
            if ip is None:
                missing.append(lp.stem)
                continue
            img = cv2.imread(str(ip))
            H, W = img.shape[:2]
            s = args.full_width / W
            full = b64jpg(cv2.resize(img, (args.full_width, int(H * s))))
            all_boxes = [[float(v) for v in p[1:5]] for _, p in sdzi]
            for i, p in sdzi:
                xc, yc, w, h = map(float, p[1:5])
                x1, y1, x2, y2 = int((xc - w / 2) * W), int((yc - h / 2) * H), int((xc + w / 2) * W), int((yc + h / 2) * H)
                m = int(0.6 * max(x2 - x1, y2 - y1)) + 40
                cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
                crop = img[cy1:cy2, cx1:cx2].copy()
                cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
                cs = 520 / max(crop.shape[:2])
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * cs)), max(1, int(crop.shape[0] * cs))),
                                  interpolation=cv2.INTER_CUBIC)
                items.append({"split": split, "frame": lp.stem, "line": i, "xc": xc, "yc": yc, "w": w, "h": h,
                              "area_pct": round(100 * w * h, 3), "full": full, "crop": b64jpg(crop),
                              "others": all_boxes})

    if not items:
        raise SystemExit("No unreviewed SDZI boxes found.")
    frames = len({(it["split"], it["frame"]) for it in items})
    html = TEMPLATE.replace("__DATA__", json.dumps(items)).replace("__TITLE__", f"{len(items)} cajas SDZI en {frames} frames")
    args.out.write_text(html, encoding="utf-8")
    size = args.out.stat().st_size / 1e6
    print(f"{len(items)} SDZI boxes in {frames} frames -> {args.out} ({size:.1f} MB)")
    if missing:
        print(f"WARNING: {len(missing)} label files without image, skipped (e.g. {missing[:3]})")


TEMPLATE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revisión SDZI</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--fg:#e8e8e8;--muted:#9aa0a6;--accent:#f5c518;--mach:#ff8c00}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{padding:10px 16px;background:var(--panel);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header b{color:var(--accent)} .bar{flex:1;min-width:160px;height:8px;background:#333;border-radius:4px;overflow:hidden}
.bar div{height:100%;background:var(--accent)} button{background:#2b3038;color:var(--fg);border:1px solid #444;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:14px}
button:hover{border-color:var(--accent)} button.sel{background:var(--accent);color:#000}
main{display:flex;gap:12px;padding:12px 16px;align-items:flex-start} .col{background:var(--panel);border-radius:8px;padding:8px;min-width:0} main .col:first-child{flex:3} main .col:last-child{flex:2} @media (max-width:700px){main{flex-direction:column}}
canvas{width:100%;display:block;cursor:default} canvas.draw{cursor:crosshair} #crop{width:100%;max-height:70vh;object-fit:contain;display:block}
.keys{display:flex;gap:8px;flex-wrap:wrap;padding:0 16px 8px} .info{padding:0 16px;color:var(--muted)} .drawing{color:var(--mach);font-weight:600}
</style></head><body>
<header><span>Revisión SDZI — __TITLE__</span><span id="pos"></span><div class="bar"><div id="prog"></div></div>
<button id="exp">Exportar JSON</button></header>
<div class="info" id="meta"></div>
<div class="keys" id="keys"></div>
<div class="info" id="hint">m maquinaria · s SDZI zona · a ambos (dibujar máquinas) · v vehículo · e edificio/cambuche · x error · ← atrás · → saltar</div>
<main><div class="col"><canvas id="full"></canvas></div><div class="col"><img id="crop" alt="zoom"></div></main>
<script>
const DATA = __DATA__;
const LABELS = {m:"maquinaria", s:"sdzi", a:"ambos", v:"vehiculo", e:"edificio", x:"error"};
const KEY = "rev_sdzi_" + DATA.length + "_" + DATA[0].frame;
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
const id = it => it.split + "/" + it.frame + "/" + it.line;
let k = DATA.findIndex(it => !dec[id(it)]); if (k < 0) k = 0;
let drawing = false, boxes = [], start = null, cur = null, img = new Image();
const cv = document.getElementById("full"), cx = cv.getContext("2d");
function save(){ try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch(e) {} }
const keysDiv = document.getElementById("keys");
for (const [kk, lab] of Object.entries(LABELS)) { const b = document.createElement("button"); b.textContent = kk + " · " + lab; b.dataset.k = kk; b.onclick = () => press(kk); keysDiv.appendChild(b); }
function draw(){
  const it = DATA[k]; cx.drawImage(img, 0, 0); const W = cv.width, H = cv.height;
  cx.lineWidth = 2; cx.strokeStyle = "#ff00ff";
  for (const o of it.others) cx.strokeRect((o[0]-o[2]/2)*W, (o[1]-o[3]/2)*H, o[2]*W, o[3]*H);
  cx.lineWidth = 4; cx.strokeStyle = "#f5c518"; cx.strokeRect((it.xc-it.w/2)*W, (it.yc-it.h/2)*H, it.w*W, it.h*H);
  const live = boxes.slice(); if (start && cur) live.push([Math.min(start[0],cur[0]),Math.min(start[1],cur[1]),Math.max(start[0],cur[0]),Math.max(start[1],cur[1])]);
  const prev = (dec[id(it)] && dec[id(it)].machinery_boxes) ? dec[id(it)].machinery_boxes : [];
  cx.lineWidth = 3; cx.strokeStyle = "#ff8c00";
  for (const b of (drawing ? live : prev)) cx.strokeRect(b[0]*W, b[1]*H, (b[2]-b[0])*W, (b[3]-b[1])*H);
}
function show(){
  const it = DATA[k]; drawing = false; boxes = []; cv.classList.remove("draw");
  img = new Image(); img.onload = () => { cv.width = img.width; cv.height = img.height; draw(); }; img.src = it.full;
  document.getElementById("crop").src = it.crop;
  const n = Object.keys(dec).length;
  document.getElementById("pos").textContent = (k+1) + " / " + DATA.length;
  document.getElementById("prog").style.width = (100*n/DATA.length) + "%";
  const d = dec[id(it)];
  document.getElementById("meta").innerHTML = it.split + "/" + it.frame + " · caja " + it.line + " · área " + it.area_pct.toFixed(2) + "% · decididas " + n + "/" + DATA.length + (d ? " · <b>actual: " + d.decision + "</b>" : "");
  for (const b of keysDiv.children) b.classList.toggle("sel", !!d && LABELS[b.dataset.k] === d.decision);
  document.getElementById("hint").className = "info"; document.getElementById("hint").textContent = "m maquinaria · s SDZI zona · a ambos (dibujar máquinas) · v vehículo · e edificio/cambuche · x error · ← atrás · → saltar";
}
function next(){ if (k < DATA.length - 1) { k++; show(); } else { show(); alert("Fin de la lista. Exporta el JSON."); } }
function press(kk){
  const it = DATA[k];
  if (kk === "a") { drawing = true; boxes = []; cv.classList.add("draw");
    const h = document.getElementById("hint"); h.className = "info drawing"; h.textContent = "DIBUJA las máquinas en la imagen izquierda (arrastrar) · z borrar última · Enter listo · Esc cancelar"; draw(); return; }
  dec[id(it)] = {decision: LABELS[kk], machinery_boxes: []}; save(); next();
}
function finishDrawing(){ const it = DATA[k]; dec[id(it)] = {decision: "ambos", machinery_boxes: boxes.slice()}; save(); drawing = false; next(); }
function pos(e){ const r = cv.getBoundingClientRect(); return [Math.min(Math.max((e.clientX-r.left)/r.width,0),1), Math.min(Math.max((e.clientY-r.top)/r.height,0),1)]; }
cv.addEventListener("mousedown", e => { if (drawing) { start = pos(e); cur = null; } });
cv.addEventListener("mousemove", e => { if (drawing && start) { cur = pos(e); draw(); } });
window.addEventListener("mouseup", e => { if (drawing && start && cur) { const b = [Math.min(start[0],cur[0]),Math.min(start[1],cur[1]),Math.max(start[0],cur[0]),Math.max(start[1],cur[1])]; if (b[2]-b[0] > 0.004 && b[3]-b[1] > 0.004) boxes.push(b); } start = cur = null; if (drawing) draw(); });
document.addEventListener("keydown", e => {
  if (drawing) { if (e.key === "Enter") finishDrawing(); else if (e.key === "z") { boxes.pop(); draw(); } else if (e.key === "Escape") show(); return; }
  if (LABELS[e.key]) press(e.key);
  else if (e.key === "ArrowLeft" || e.key === "b") { if (k > 0) { k--; show(); } }
  else if (e.key === "ArrowRight") next();
});
document.getElementById("exp").onclick = () => {
  const out = DATA.filter(it => dec[id(it)]).map(it => { const d = dec[id(it)];
    return {split: it.split, frame: it.frame, line: it.line, xc: it.xc, yc: it.yc, w: it.w, h: it.h, area_pct: it.area_pct, decision: d.decision,
      machinery_boxes: d.machinery_boxes.map(b => [((b[0]+b[2])/2).toFixed(6), ((b[1]+b[3])/2).toFixed(6), (b[2]-b[0]).toFixed(6), (b[3]-b[1]).toFixed(6)].join(" ")).join("|")}; });
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "revision_sdzi_" + out.length + ".json"; a.click();
};
show();
</script></body></html>"""

if __name__ == "__main__":
    main()
