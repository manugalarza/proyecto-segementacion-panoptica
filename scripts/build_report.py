"""Build reporte_entrenamiento/index.html: training time, epochs, loss curves and
(optional) evaluation metrics for every run folder passed with --run.

Reads, per run folder: train_log.csv (epoch, mean_loss, stuff_loss, thing_loss, minutes)
and run_info.json. Optional --metrics JSON files (from eval scripts) are added as tables.

Example (Windows, repo root):
    .venv\\Scripts\\python scripts\\build_report.py --run A=runs\\A --run v2_A=checkpoints_v2\\A ^
        --out reporte_entrenamiento
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(folder: Path) -> dict:
    rows = list(csv.DictReader(open(folder / "train_log.csv", encoding="utf-8")))
    info = json.load(open(folder / "run_info.json", encoding="utf-8")) if (folder / "run_info.json").exists() else {}
    minutes = [float(r["minutes"]) for r in rows]
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "mean": [float(r["mean_loss"]) for r in rows],
        "stuff": [float(r["stuff_loss"]) for r in rows],
        "thing": [float(r["thing_loss"]) for r in rows],
        "minutes": minutes,
        "info": info,
    }


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def curves_png(runs: dict[str, dict]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in runs.items():
        axes[0].plot(r["epoch"], r["stuff"], label=name)
        axes[1].plot(r["epoch"], r["thing"], label=name)
    axes[0].set_title("Pérdida de stuff (entrenamiento)")
    axes[1].set_title("Pérdida de things (entrenamiento)")
    for ax in axes:
        ax.set_xlabel("época")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    return fig_to_b64(fig)


def html_table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=FOLDER (repeatable)")
    ap.add_argument("--metrics", action="append", default=[], help="JSON with evaluation metrics (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vis-dir", type=Path, default=None, help="Folder with eval_pixel --save-vis images (vis/)")
    ap.add_argument("--vis-n", type=int, default=6, help="How many visualization frames to embed")
    ap.add_argument("--things-vis-dir", type=Path, default=None, help="eval_things --save-vis folder (one subfolder per model)")
    ap.add_argument("--things-vis-n", type=int, default=29, help="Centroid frames to embed per model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        runs[name] = load_run(Path(path))

    timing_rows, meta_rows = [], []
    for name, r in runs.items():
        info = r["info"]
        n = len(r["epoch"])
        per_epoch = sum(r["minutes"]) / n if n else 0
        timing_rows.append([name, info.get("device", "?"), info.get("encoder", "?"), info.get("stuff_target", "?"),
                            info.get("train_frames", "?"), n, info.get("epochs", "?"),
                            f"{per_epoch:.2f}", f"{sum(r['minutes']) / 60:.2f}",
                            f"{r['mean'][-1]:.5f}" if n else "-", info.get("seed", "?")])
        meta_rows.append([name, info.get("data", "?")])
    timing_html = html_table(["Corrida", "Hardware", "Modelo", "Target de stuff", "Frames train", "Épocas hechas",
                              "Épocas planeadas", "Min/época", "Total (h)", "Pérdida final", "Semilla"],
                             timing_rows)
    data_html = html_table(["Corrida", "Carpeta de datos"], meta_rows)

    metric_html = ""
    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from flatten(v, f"{prefix} / {k}" if prefix else str(k))
        elif isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj) and len(obj) > 0 and "frame" not in obj[0]:
                for i, x in enumerate(obj):
                    yield from flatten(x, f"{prefix}[{i}]")
            else:
                yield prefix, f"{len(obj)} elementos"
        else:
            yield prefix, obj

    def pct(x):
        return "-" if x is None else f"{float(x):.2f}"

    def things_table(m: dict) -> str:
        rows = []
        for key, classes in m.items():
            if not isinstance(classes, dict):
                continue
            model, _, thr = key.partition("@")
            for cls, c in classes.items():
                gt, found, peaks, correct = c["gt"], c["found"], c["peaks"], c["correct"]
                rows.append([model, thr, cls, gt, found, pct(found / gt if gt else None),
                             peaks, correct, pct(correct / peaks if peaks else None)])
        return html_table(["Modelo", "Umbral", "Clase", "Regiones GT", "Encontradas", "Recall",
                           "Picos", "Correctos", "Precisión"], rows)

    def pixel_table(m: dict) -> str:
        rows = []
        for model, r in m.get("models", {}).items():
            for cls in ("river", "road"):
                if f"{cls}_iou" not in r:
                    continue
                ci = r[f"{cls}_recall_cov_ci95"]
                rows.append([model, cls, pct(r[f"{cls}_iou"]), pct(r[f"{cls}_pixel_precision"]),
                             pct(r[f"{cls}_pixel_recall"]), r[f"{cls}_regions_found"],
                             pct(r[f"{cls}_recall_cov"]), f"[{pct(ci[0])}, {pct(ci[1])}]",
                             pct(r[f"{cls}_precision_cov"]), r[f"{cls}_pred_regions_correct"]])
        return html_table(["Modelo", "Clase", "IoU", "Precisión píxel", "Recall píxel", "Regiones GT encontradas",
                           "Recall por región", "IC 95%", "Precisión por región", "Regiones predichas correctas"], rows)

    for mpath in args.metrics:
        m = json.load(open(mpath, encoding="utf-8"))
        if "models" in m and isinstance(m["models"], dict):
            s_ = m.get("settings", {})
            metric_html += (f"<h3>Evaluación de píxeles (gt_manual, {len(s_.get('frames', []))} frames)</h3>"
                            "<p>Región encontrada si la predicción cubre al menos el 30% de sus píxeles. "
                            "Río y carretera; SDZI queda ignorado.</p>") + pixel_table(m)
        elif all(isinstance(v, dict) and "vehicle" in v or "building" in v for v in m.values()):
            metric_html += ("<h3>Evaluación de things (centros de calor, tolerancia 15 px, gt_manual)</h3>"
                            "<p>Recall = regiones GT con un pico de la misma clase dentro. "
                            "Precisión = picos correctos / picos.</p>") + things_table(m)
        else:
            rows = [[k, v] for k, v in flatten(m) if k not in ("frames",)]
            metric_html += f"<h3>{Path(mpath).name}</h3>" + html_table(["métrica", "valor"], rows)
    if not metric_html:
        metric_html = ("<p><em>Pendiente: métricas de evaluación (things y stuff con la taxonomía nueva). "
                       "Se agregan al correr eval_things.py y eval_pixel.py con --metrics.</em></p>")

    img = curves_png(runs)
    vis_html = ""
    if args.vis_dir and args.vis_dir.is_dir():
        files = sorted(args.vis_dir.glob("*.jpg")) + sorted(args.vis_dir.glob("*.png"))
        if files:
            step = max(1, len(files) // args.vis_n)
            picked = files[::step][: args.vis_n]
            vis_html = "<p>Imagen | ground truth | un panel por modelo (río cian, carretera amarilla, gris = ignorado).</p>"
            for f in picked:
                mime = "image/jpeg" if f.suffix == ".jpg" else "image/png"
                b64 = base64.b64encode(f.read_bytes()).decode()
                vis_html += f'<p><strong>{f.stem}</strong></p><img alt="{f.stem}" src="data:{mime};base64,{b64}">'
    if not vis_html:
        vis_html = "<p><em>Pendiente: correr eval_pixel.py con --save-vis.</em></p>"
    if args.things_vis_dir and args.things_vis_dir.is_dir():
        tv = "<h3>Centroides de things</h3><p>Contorno = ground truth; punto = pico predicho (umbral 0,3 y 0,1 según la carpeta).</p>"
        for model_dir in sorted(p for p in args.things_vis_dir.iterdir() if p.is_dir()):
            files = sorted(model_dir.glob("*.jpg")) + sorted(model_dir.glob("*.png"))
            if not files:
                continue
            step = max(1, len(files) // args.things_vis_n)
            tv += f"<h4>{model_dir.name}</h4>"
            for f in files[::step][: args.things_vis_n]:
                mime = "image/jpeg" if f.suffix == ".jpg" else "image/png"
                b64 = base64.b64encode(f.read_bytes()).decode()
                tv += f'<p><strong>{f.stem}</strong></p><img alt="{f.stem}" src="data:{mime};base64,{b64}">'
        metric_html += tv
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Reporte de entrenamiento</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#222;background:#fff}}
h1{{margin-bottom:4px}} h2{{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}} th{{background:#f4f4f4}}
img{{max-width:100%}} .note{{background:#fff8e1;border-left:4px solid #f0b400;padding:8px 12px;font-size:14px}}
</style></head><body>
<h1>Reporte de entrenamiento: Panoptic FCN</h1>
<p>Generado por scripts/build_report.py. Todas las cifras salen de los logs de cada corrida.</p>
<h2>1. Tiempo y épocas</h2>{timing_html}
<h2>2. Datos por corrida</h2>{data_html}
<h2>3. Curvas de aprendizaje (entrenamiento)</h2>
<img alt="curvas" src="data:image/png;base64,{img}">
<h2>4. Métricas de evaluación</h2>{metric_html}
<h2>5. Imágenes de evolución</h2>{vis_html}
</body></html>"""
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(f"Reporte: {args.out / 'index.html'}")
    print(f"Corridas: {', '.join(runs)}")


if __name__ == "__main__":
    main()
