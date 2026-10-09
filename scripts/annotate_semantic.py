"""
annotate_semantic.py
--------------------
Desktop GUI to paint SEMANTIC segmentation masks (one class per pixel).
Same spirit and folder convention as annotate_images.py (the YOLO box tool),
but the output is a PNG mask per image instead of a .txt with boxes:

    dataset/
    ├── images/            <- images to annotate
    ├── masks/             <- <stem>.png written here (auto-created)
    └── semantic.yaml      <- class names of the masks (auto-created / read)

Mask format (masks/<stem>.png, 8-bit single channel, same size as the image):
    pixel value = class id (0..N-1), 255 = unlabeled / ignore
Default class ids match the YOLO ids of the project so boxes and masks agree:
    0 vehicle, 1 building, 2 road, 3 river, 4 SDZI, 5 background
dataset.yaml (YOLO) is never touched.

Tools
  B  brush       drag to paint with the active class
  P  polygon     click vertices; Enter / double-click closes, Esc cancels,
                 Backspace removes the last vertex
  S  SAM box     drag a box -> SAM mask is painted with the active class
                 (only with --sam MODEL, needs ultralytics). Always review it.
  1-9            active class          0  eraser (paints "unlabeled")
  [ ]            brush size            K  protect: only paint unlabeled pixels
  F  fill all unlabeled pixels with "background"
  V  show / hide the mask              O  cycle mask opacity
  G  show / hide reference boxes (--boxes)
  Ctrl+Z undo     Left/Right previous/next image (auto-saves)    Ctrl+S save
  Mouse wheel zoom at cursor, right-drag (or middle-drag) pans, R resets view

Workflow tip: paint the objects first, then F to make the rest background,
and check that "unlabeled" in the sidebar is 0% before moving on.

Dependencies:  pip install Pillow PyYAML numpy      (SAM: pip install ultralytics)

Usage:
    python scripts/annotate_semantic.py --dataset ./eval_subset
    python scripts/annotate_semantic.py --images ./eval_subset/images
    # optional: show YOLO boxes as a dashed guide, start from existing masks
    #           (e.g. SAM pseudo-masks), enable the SAM box tool
    python scripts/annotate_semantic.py --dataset ./eval_subset ^
        --boxes ./datos_2x2/abril/test/labels --init ./sam_masks --sam sam_b.pt --device 0
"""

from __future__ import annotations

import argparse
import re
import tkinter as tk
from pathlib import Path
from tkinter import Button, Canvas, Frame, IntVar, Label, StringVar, Tk, messagebox, ttk

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    raise SystemExit("Pillow and numpy are required.  Run:  pip install Pillow numpy")
try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is required.  Run:  pip install PyYAML")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = ["vehicle", "building", "road", "river", "SDZI", "background"]
BACKGROUND_NAMES = {"background", "none", "fondo"}
UNLABELED = 255

# RGB; first five match draw_yolo_labels.py (vehicle red, building blue,
# road yellow, river cyan, SDZI magenta), background grey.
CLASS_COLORS = [
    (255, 0, 0), (0, 128, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 128, 128), (76, 175, 80), (255, 152, 0), (156, 39, 176),
]
OPACITIES = [0.35, 0.55, 0.75]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UNDO_LIMIT = 25
UI_BG, PANEL_BG, BTN_BG, ACCENT = "#1a1a2e", "#16213e", "#0f3460", "#e94560"


def hex_color(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def natural_sort_key(path: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", Path(path).name)]


# ---------------------------------------------------------------------------
# Paths / yaml
# ---------------------------------------------------------------------------

def resolve_paths(dataset_root, images_dir, masks_dir):
    if dataset_root:
        root = Path(dataset_root).resolve()
        images = root / "images"
    else:
        images = Path(images_dir).resolve()
        root = images.parent
    masks = Path(masks_dir).resolve() if masks_dir else root / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    return images, masks, root / "semantic.yaml"


def load_classes(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        return []
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in names] if isinstance(names, list) else []


def save_classes(yaml_path: Path, class_names: list[str]):
    data = {
        "format": "PNG masks, pixel = class id, 255 = unlabeled/ignore",
        "nc": len(class_names),
        "names": {i: n for i, n in enumerate(class_names)},
        "ignore_index": UNLABELED,
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_yolo_boxes(path: Path, w: int, h: int):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                c, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                boxes.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


# ---------------------------------------------------------------------------
# Mask store
# ---------------------------------------------------------------------------

class MaskStore:
    def __init__(self, masks_dir: Path, init_dir: Path | None):
        self.masks_dir = masks_dir
        self.init_dir = init_dir

    def path(self, img_path: str) -> Path:
        return self.masks_dir / (Path(img_path).stem + ".png")

    def load(self, img_path: str, w: int, h: int) -> tuple[np.ndarray, str]:
        for source, p in (("saved", self.path(img_path)),
                          ("init", self.init_dir / (Path(img_path).stem + ".png") if self.init_dir else None)):
            if p is not None and p.exists():
                m = Image.open(p)
                if m.mode not in ("L", "P"):
                    m = m.convert("L")
                if m.size != (w, h):
                    m = m.resize((w, h), Image.NEAREST)
                return np.array(m, dtype=np.uint8), source
        return np.full((h, w), UNLABELED, dtype=np.uint8), "new"

    def save(self, img_path: str, mask: np.ndarray):
        Image.fromarray(mask, mode="L").save(self.path(img_path))

    def is_done(self, img_path: str) -> bool:
        return self.path(img_path).exists()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class SemanticAnnotator:
    def __init__(self, root: Tk, images_path: Path, masks_path: Path, yaml_path: Path,
                 class_names: list[str], boxes_dir: Path | None, init_dir: Path | None,
                 sam_model: str | None, device: str):
        self.root = root
        self.images_path, self.masks_path, self.yaml_path = images_path, masks_path, yaml_path
        self.class_names = class_names[:9]
        self.colors = CLASS_COLORS[: len(self.class_names)]
        self.store = MaskStore(masks_path, init_dir)
        self.boxes_dir = boxes_dir
        self.sam_model_name, self.device, self._sam = sam_model, device, None

        self.images = sorted((str(p) for p in images_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS),
                             key=natural_sort_key)
        if not self.images:
            messagebox.showerror("No images", f"No images found in:\n{images_path}")
            root.destroy()
            return

        # LUT for coloring the mask (unlabeled stays transparent)
        self.lut = np.zeros((256, 3), dtype=np.float32)
        for i, c in enumerate(self.colors):
            self.lut[i] = c

        self.img_index = 0
        self.image_np: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.dirty = False
        self.undo_stack: list[np.ndarray] = []

        self.active_class = IntVar(value=0)
        self.brush = 20                 # radius in image pixels
        self.protect = False
        self.show_mask = True
        self.show_boxes = True
        self.opacity_i = 1
        self.mode = "brush"             # brush | polygon | sam
        self.zoom, self.fit, self.ox, self.oy = 1.0, 1.0, 0.0, 0.0

        self._stroke: list[tuple[float, float]] = []
        self._poly: list[tuple[float, float]] = []
        self._pan = None
        self._sam_start = None
        self._cursor_item = None
        self._tk_img = None

        self._build_ui()
        self._bind()
        self.root.after(50, lambda: self._load_image(0, save=False))

    # ---------------- UI ----------------
    def _build_ui(self):
        self.root.title("Semantic Mask Annotator")
        self.root.configure(bg=UI_BG)
        btn = {"bg": BTN_BG, "fg": "#e0e0e0", "relief": "flat", "padx": 10, "pady": 4, "cursor": "hand2",
               "activebackground": ACCENT, "activeforeground": "white", "font": ("Courier", 9, "bold")}

        bar = Frame(self.root, bg=PANEL_BG, pady=6)
        bar.pack(fill="x", side="top")
        Button(bar, text="◀ Prev", command=self._prev, **btn).pack(side="left", padx=3)
        Button(bar, text="Next ▶", command=self._next, **btn).pack(side="left", padx=3)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        self.mode_buttons = {}
        for m, txt in (("brush", "🖌 Brush [B]"), ("polygon", "⬠ Polygon [P]"), ("sam", "▣ SAM box [S]")):
            b = Button(bar, text=txt, command=lambda m=m: self._set_mode(m), **btn)
            b.pack(side="left", padx=2)
            self.mode_buttons[m] = b
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        Button(bar, text="↶ Undo", command=self._undo, **btn).pack(side="left", padx=2)
        Button(bar, text="Fill bg [F]", command=self._fill_background, **btn).pack(side="left", padx=2)
        Button(bar, text="🗑 Clear", command=self._clear, **btn).pack(side="left", padx=2)
        Button(bar, text="💾 Save", command=self._save, **btn).pack(side="left", padx=2)
        self.counter_var = StringVar()
        Label(bar, textvariable=self.counter_var, bg=PANEL_BG, fg="#a8dadc",
              font=("Courier", 10)).pack(side="right", padx=10)

        # bottom bars first so they span the full width
        self.status_var = StringVar()
        sb = Frame(self.root, bg=BTN_BG)
        sb.pack(fill="x", side="bottom")
        Label(sb, textvariable=self.status_var, bg=BTN_BG, fg="#a8dadc", font=("Courier", 8),
              anchor="w", padx=8).pack(fill="x")
        pf = Frame(self.root, bg=PANEL_BG, pady=4)
        pf.pack(fill="x", side="bottom")
        Label(pf, text="Progress:", bg=PANEL_BG, fg="#a8dadc", font=("Courier", 8)).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(pf, length=300, mode="determinate")
        self.progress.pack(side="left")
        self.progress_var = StringVar()
        Label(pf, textvariable=self.progress_var, bg=PANEL_BG, fg="#a8dadc", font=("Courier", 8)).pack(side="left", padx=6)

        side = Frame(self.root, bg=PANEL_BG, width=200, padx=10, pady=10)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        Label(side, text="CLASSES", bg=PANEL_BG, fg=ACCENT, font=("Courier", 10, "bold")).pack(anchor="w", pady=(0, 6))
        for i, name in enumerate(self.class_names):
            tk.Radiobutton(side, text=f"[{i + 1}] {name}", variable=self.active_class, value=i,
                           bg=PANEL_BG, fg=hex_color(self.colors[i]), selectcolor=BTN_BG,
                           activebackground=PANEL_BG, font=("Courier", 9, "bold"), anchor="w",
                           command=self._update_info).pack(fill="x")
        tk.Radiobutton(side, text="[0] eraser", variable=self.active_class, value=UNLABELED,
                       bg=PANEL_BG, fg="white", selectcolor=BTN_BG, activebackground=PANEL_BG,
                       font=("Courier", 9, "bold"), anchor="w", command=self._update_info).pack(fill="x")
        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        self.tool_var = StringVar()
        Label(side, textvariable=self.tool_var, bg=PANEL_BG, fg="white", font=("Courier", 8),
              justify="left").pack(anchor="w")
        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        Label(side, text="PIXELS (% image)", bg=PANEL_BG, fg="#a8dadc", font=("Courier", 8)).pack(anchor="w")
        self.stats_var = StringVar()
        Label(side, textvariable=self.stats_var, bg=PANEL_BG, fg="white", font=("Courier", 8),
              justify="left").pack(anchor="w")

        cf = Frame(self.root, bg=UI_BG)
        cf.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.canvas = Canvas(cf, bg="#0d0d1a", cursor="crosshair", highlightthickness=1,
                             highlightbackground=ACCENT)
        self.canvas.pack(fill="both", expand=True)

    def _bind(self):
        c, r = self.canvas, self.root
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", lambda e: self._close_polygon())
        c.bind("<Motion>", self._on_motion)
        for b in ("2", "3"):
            c.bind(f"<ButtonPress-{b}>", self._pan_start)
            c.bind(f"<B{b}-Motion>", self._pan_move)
        c.bind("<MouseWheel>", lambda e: self._zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8))
        c.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))   # Linux
        c.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))
        c.bind("<Configure>", lambda e: self._fit_view())

        r.bind("<Left>", lambda e: self._prev())
        r.bind("<Right>", lambda e: self._next())
        r.bind("<Control-z>", lambda e: self._undo())
        r.bind("<Control-s>", lambda e: self._save())
        r.bind("<Return>", lambda e: self._close_polygon())
        r.bind("<Escape>", lambda e: self._cancel_polygon())
        r.bind("<BackSpace>", lambda e: self._pop_vertex())
        for i in range(len(self.class_names)):
            r.bind(str(i + 1), lambda e, ci=i: self._pick(ci))
        r.bind("0", lambda e: self._pick(UNLABELED))
        keys = {"b": lambda: self._set_mode("brush"), "p": lambda: self._set_mode("polygon"),
                "s": lambda: self._set_mode("sam"), "f": self._fill_background, "k": self._toggle_protect,
                "v": self._toggle_mask, "g": self._toggle_boxes, "o": self._cycle_opacity,
                "r": self._fit_view, "bracketleft": lambda: self._brush_size(1 / 1.3),
                "bracketright": lambda: self._brush_size(1.3)}
        for k, fn in keys.items():
            r.bind(f"<{k}>", lambda e, fn=fn: fn())
            if len(k) == 1:
                r.bind(f"<{k.upper()}>", lambda e, fn=fn: fn())

    # ---------------- image / view ----------------
    def _load_image(self, index: int, save=True):
        if save:
            self._save(silent=True)
        self.img_index = index
        path = self.images[index]
        img = Image.open(path).convert("RGB")
        self.image_pil = img
        self.image_np = np.asarray(img, dtype=np.float32)
        self.h, self.w = self.image_np.shape[:2]
        self.mask, source = self.store.load(path, self.w, self.h)
        self.dirty = source == "init"     # an init mask becomes a saved mask once you move on
        self.undo_stack.clear()
        self._poly, self._stroke = [], []
        self.boxes = read_yolo_boxes(self.boxes_dir / (Path(path).stem + ".txt"), self.w, self.h) if self.boxes_dir else []
        self._fit_view()
        self.counter_var.set(f"Image {index + 1}/{len(self.images)} │ {Path(path).name}")
        self._update_progress()
        self._status(f"Loaded {Path(path).name} (mask: {source})")

    def _fit_view(self):
        if self.image_np is None:
            return
        self.canvas.update_idletasks()
        cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        self.fit = min(cw / self.w, ch / self.h)
        self.zoom = 1.0
        s = self.fit
        self.ox, self.oy = (cw - self.w * s) / 2, (ch - self.h * s) / 2
        self._render()

    @property
    def scale(self):
        return self.fit * self.zoom

    def _c2i(self, cx, cy):
        return (cx - self.ox) / self.scale, (cy - self.oy) / self.scale

    def _i2c(self, ix, iy):
        return ix * self.scale + self.ox, iy * self.scale + self.oy

    def _zoom_at(self, cx, cy, factor):
        new = min(max(self.zoom * factor, 1.0), 40.0)
        ix, iy = self._c2i(cx, cy)
        self.zoom = new
        self.ox, self.oy = cx - ix * self.scale, cy - iy * self.scale
        self._render()

    def _pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy)

    def _pan_move(self, e):
        if self._pan:
            x0, y0, ox, oy = self._pan
            self.ox, self.oy = ox + e.x - x0, oy + e.y - y0
            self._render()

    def _render(self):
        if self.image_np is None:
            return
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        s = self.scale
        # visible region in image coordinates
        x0 = int(max(0, np.floor(-self.ox / s)))
        y0 = int(max(0, np.floor(-self.oy / s)))
        x1 = int(min(self.w, np.ceil((cw - self.ox) / s)))
        y1 = int(min(self.h, np.ceil((ch - self.oy) / s)))
        if x1 > x0 and y1 > y0:
            dw, dh = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
            box = (x0, y0, x1, y1)
            img = np.asarray(self.image_pil.crop(box).resize((dw, dh), Image.BILINEAR if s < 1 else Image.NEAREST),
                             dtype=np.float32)
            if self.show_mask:
                m = np.asarray(Image.fromarray(self.mask[y0:y1, x0:x1]).resize((dw, dh), Image.NEAREST))
                a = OPACITIES[self.opacity_i]
                lab = (m != UNLABELED)[..., None]
                img = np.where(lab, img * (1 - a) + self.lut[m] * a, img)
            self._tk_img = ImageTk.PhotoImage(Image.fromarray(img.astype(np.uint8)))
            c.create_image(self.ox + x0 * s, self.oy + y0 * s, anchor="nw", image=self._tk_img)
        if self.show_boxes:
            for cls, bx1, by1, bx2, by2 in self.boxes:
                color = hex_color(CLASS_COLORS[cls]) if 0 <= cls < len(CLASS_COLORS) else "white"
                c.create_rectangle(*self._i2c(bx1, by1), *self._i2c(bx2, by2), outline=color, width=1, dash=(5, 3))
        if self._poly:
            pts = [p for v in self._poly for p in self._i2c(*v)]
            color = self._active_color()
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=2)
            for v in self._poly:
                vx, vy = self._i2c(*v)
                c.create_oval(vx - 3, vy - 3, vx + 3, vy + 3, fill=color, outline="white")
        self._cursor_item = None
        self._update_info()

    # ---------------- editing ----------------
    def _active_color(self):
        ci = self.active_class.get()
        return "white" if ci == UNLABELED else hex_color(self.colors[ci])

    def _push_undo(self):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)

    def _apply(self, region: np.ndarray):
        """Paint the active class on the boolean region (respects protect)."""
        ci = self.active_class.get()
        if self.protect and ci != UNLABELED:
            region = region & (self.mask == UNLABELED)
        if not region.any():
            return
        self._push_undo()
        self.mask[region] = ci
        self.dirty = True
        self._render()

    def _shape_region(self, draw_fn) -> np.ndarray:
        layer = Image.new("L", (self.w, self.h), 0)
        draw_fn(ImageDraw.Draw(layer))
        return np.asarray(layer) > 0

    def _on_press(self, e):
        ix, iy = self._c2i(e.x, e.y)
        if self.mode == "brush":
            self._stroke = [(ix, iy)]
        elif self.mode == "polygon":
            self._poly.append((min(max(ix, 0), self.w), min(max(iy, 0), self.h)))
            self._render()
        elif self.mode == "sam":
            self._sam_start = (e.x, e.y)

    def _on_drag(self, e):
        ix, iy = self._c2i(e.x, e.y)
        if self.mode == "brush" and self._stroke:
            px, py = self._i2c(*self._stroke[-1])
            self._stroke.append((ix, iy))
            self.canvas.create_line(px, py, e.x, e.y, fill=self._active_color(),
                                    width=max(1, 2 * self.brush * self.scale), capstyle="round", tags="live")
        elif self.mode == "sam" and self._sam_start:
            self.canvas.delete("live")
            self.canvas.create_rectangle(*self._sam_start, e.x, e.y, outline=self._active_color(),
                                         width=2, dash=(4, 2), tags="live")
        self._on_motion(e)

    def _on_release(self, e):
        if self.mode == "brush" and self._stroke:
            pts, r = self._stroke, self.brush

            def draw(d):
                if len(pts) > 1:
                    d.line(pts, fill=255, width=int(2 * r), joint="curve")
                for x, y in (pts if len(pts) < 50 else pts[:: max(1, len(pts) // 50)] + [pts[-1]]):
                    d.ellipse((x - r, y - r, x + r, y + r), fill=255)
            self._stroke = []
            self._apply(self._shape_region(draw))
        elif self.mode == "sam" and self._sam_start:
            x0, y0 = self._c2i(*self._sam_start)
            x1, y1 = self._c2i(e.x, e.y)
            self._sam_start = None
            self.canvas.delete("live")
            bx = [max(0, min(x0, x1)), max(0, min(y0, y1)), min(self.w, max(x0, x1)), min(self.h, max(y0, y1))]
            if bx[2] - bx[0] > 4 and bx[3] - bx[1] > 4:
                self._run_sam(bx)

    def _close_polygon(self):
        if self.mode != "polygon" or len(self._poly) < 3:
            return
        pts = list(self._poly)
        self._poly = []
        self._apply(self._shape_region(lambda d: d.polygon(pts, fill=255)))
        self._render()

    def _cancel_polygon(self):
        self._poly = []
        self._render()

    def _pop_vertex(self):
        if self._poly:
            self._poly.pop()
            self._render()

    def _run_sam(self, box):
        if not self.sam_model_name:
            self._status("SAM disabled: start the tool with --sam sam_b.pt (or sam2.1_b.pt)")
            return
        try:
            if self._sam is None:
                self._status("Loading SAM...")
                self.root.update_idletasks()
                from ultralytics import SAM
                self._sam = SAM(self.sam_model_name)
            bgr = np.ascontiguousarray(self.image_np[..., ::-1].astype(np.uint8))
            res = self._sam(bgr, bboxes=[box], device=self.device, verbose=False)[0]
        except Exception as ex:  # noqa: BLE001
            self._status(f"SAM failed: {ex}")
            return
        if res.masks is None:
            self._status("SAM returned no mask")
            return
        m = res.masks.data[0].cpu().numpy() > 0.5
        if m.shape != (self.h, self.w):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((self.w, self.h), Image.NEAREST)) > 0
        self._apply(m)
        self._status(f"SAM mask painted ({m.sum() / ((box[2] - box[0]) * (box[3] - box[1])):.0%} of the box). Review the edges.")

    def _fill_background(self):
        bg = next((i for i, n in enumerate(self.class_names) if n.lower() in BACKGROUND_NAMES), None)
        if bg is None:
            self._status("No background class in the class list")
            return
        region = self.mask == UNLABELED
        if region.any():
            self._push_undo()
            self.mask[region] = bg
            self.dirty = True
            self._render()
            self._status("Unlabeled pixels -> background")

    def _clear(self):
        if messagebox.askyesno("Clear", "Set every pixel of this image to unlabeled?"):
            self._push_undo()
            self.mask[:] = UNLABELED
            self.dirty = True
            self._render()

    def _undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.dirty = True
            self._render()
            self._status("Undo")

    def _save(self, silent=False):
        if self.mask is None:
            return
        path = self.images[self.img_index]
        if self.dirty or not self.store.is_done(path):
            if self.dirty or (self.mask != UNLABELED).any():
                self.store.save(path, self.mask)
                self.dirty = False
        self._update_progress()
        if not silent:
            self._status(f"Saved -> {self.store.path(path)}")

    # ---------------- small actions ----------------
    def _pick(self, ci):
        self.active_class.set(ci)
        self._update_info()

    def _set_mode(self, m):
        self.mode = m
        self._poly, self._stroke = [], []
        for k, b in self.mode_buttons.items():
            b.config(bg=ACCENT if k == m else BTN_BG)
        hints = {"brush": "drag to paint, [ ] size", "polygon": "click vertices, Enter/double-click closes",
                 "sam": "drag a box around ONE object/zone"}
        self._status(f"Mode {m}: {hints[m]}")
        self._render()

    def _brush_size(self, f):
        self.brush = int(min(max(self.brush * f, 1), 400))
        self._update_info()

    def _toggle_protect(self):
        self.protect = not self.protect
        self._update_info()

    def _toggle_mask(self):
        self.show_mask = not self.show_mask
        self._render()

    def _toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self._render()

    def _cycle_opacity(self):
        self.opacity_i = (self.opacity_i + 1) % len(OPACITIES)
        self._render()

    def _prev(self):
        if self.img_index > 0:
            self._load_image(self.img_index - 1)

    def _next(self):
        if self.img_index < len(self.images) - 1:
            self._load_image(self.img_index + 1)

    def _on_motion(self, e):
        if self.mode != "brush" or self.image_np is None:
            return
        r = self.brush * self.scale
        if self._cursor_item is None:
            self._cursor_item = self.canvas.create_oval(0, 0, 0, 0, outline="white", dash=(2, 2))
        self.canvas.coords(self._cursor_item, e.x - r, e.y - r, e.x + r, e.y + r)
        self.canvas.tag_raise(self._cursor_item)

    # ---------------- info ----------------
    def _status(self, msg):
        self.status_var.set(msg)

    def _update_info(self):
        ci = self.active_class.get()
        name = "eraser" if ci == UNLABELED else self.class_names[ci]
        self.tool_var.set(f"mode    : {self.mode}\nclass   : {name}\nbrush r : {self.brush}px\n"
                          f"protect : {'ON' if self.protect else 'off'}\nzoom    : {self.zoom:.1f}x\n"
                          f"opacity : {OPACITIES[self.opacity_i]:.0%}")
        if self.mask is not None:
            counts = np.bincount(self.mask.ravel(), minlength=256)
            total = self.mask.size
            lines = [f"{n[:11]:<11} {100 * counts[i] / total:5.1f}" for i, n in enumerate(self.class_names) if counts[i]]
            lines.append(f"{'unlabeled':<11} {100 * counts[UNLABELED] / total:5.1f}")
            self.stats_var.set("\n".join(lines))

    def _update_progress(self):
        done = sum(self.store.is_done(p) for p in self.images)
        self.progress["maximum"] = len(self.images)
        self.progress["value"] = done
        self.progress_var.set(f"{done}/{len(self.images)} masks saved")

    def on_close(self):
        self._save(silent=True)
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Semantic segmentation mask painter (PNG masks)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dataset", metavar="DIR", help="Dataset root with an images/ sub-folder")
    g.add_argument("--images", metavar="DIR", help="Folder with the images")
    ap.add_argument("--masks", metavar="DIR", help="Where masks are written (default: <root>/masks)")
    ap.add_argument("--classes", nargs="*", default=[], metavar="NAME",
                    help=f"Class names in id order, up to 9 (default {' '.join(DEFAULT_CLASSES)})")
    ap.add_argument("--boxes", metavar="DIR", help="Optional YOLO labels folder drawn as a dashed guide")
    ap.add_argument("--init", metavar="DIR", help="Optional folder of starting masks (same PNG format), "
                                                   "used only when no saved mask exists")
    ap.add_argument("--sam", metavar="MODEL", help="Enable the SAM box tool, e.g. sam_b.pt or sam2.1_b.pt")
    ap.add_argument("--device", default="cpu", help="SAM device: cpu or 0 (GPU)")
    args = ap.parse_args()
    if not args.dataset and not args.images:
        args.dataset = "."

    images_path, masks_path, yaml_path = resolve_paths(args.dataset, args.images, args.masks)
    if not images_path.exists():
        raise SystemExit(f"Images folder not found: {images_path}")
    class_names = list(args.classes) or load_classes(yaml_path) or list(DEFAULT_CLASSES)
    if len(class_names) > 9:
        raise SystemExit("At most 9 classes (hotkeys 1-9)")
    existing = load_classes(yaml_path)
    if existing and existing != class_names:
        raise SystemExit(f"{yaml_path} has classes {existing}; --classes {class_names} would change the ids "
                         "of saved masks. Use another --masks folder or edit the yaml on purpose.")
    if not existing:
        save_classes(yaml_path, class_names)

    print(f"Images : {images_path}\nMasks  : {masks_path}\nYAML   : {yaml_path}\nClasses: {class_names}")
    root = Tk()
    root.geometry("1300x800")
    root.minsize(900, 600)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TProgressbar", troughcolor=BTN_BG, background=ACCENT)
    app = SemanticAnnotator(root, images_path, masks_path, yaml_path, class_names,
                            Path(args.boxes) if args.boxes else None, Path(args.init) if args.init else None,
                            args.sam, args.device)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
