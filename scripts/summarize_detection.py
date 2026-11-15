"""Prints one compact table from every detection_summary.json under a folder
(written by eval_detection.py), so the results can be pasted in one block.

Usage (repo root):
    .venv\\Scripts\\python scripts\\summarize_detection.py ..\\resultados_resolucion
"""
import json
import sys
from pathlib import Path


def f(x):
    return "  n/a" if x is None else f"{x:.3f}"


def main():
    root = Path(sys.argv[1])
    files = sorted(root.rglob("detection_summary.json"))
    if not files:
        raise SystemExit(f"No detection_summary.json under {root}")
    print(" | ".join(["run", "model", "frames", "mAP50", "CI95", "veh AP50", "bld AP50", "mac AP50",
                      "veh+bld", "bld maxR", "bld R@c", "bld P@c"]))
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        n = d["settings"].get("n_frames")
        conf = d["settings"].get("conf", 0.25)
        for name, r in d["models"].items():
            ci = r.get("ci95_AP50", {}).get("mAP50")
            ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-"
            v, b, m = (r.get(k, {}) for k in ("vehicle", "building", "machinery"))
            shared = [x for x in (v.get("AP50"), b.get("AP50")) if x is not None]
            print(" | ".join([p.parent.name, name, str(n), f(r.get("mAP50")), ci_s,
                              f(v.get("AP50")), f(b.get("AP50")), f(m.get("AP50")),
                              f(sum(shared) / len(shared)) if shared else "n/a",
                              f(b.get("max_recall")), f(b.get(f"recall@conf{conf}")), f(b.get(f"precision@conf{conf}"))]))


if __name__ == "__main__":
    main()
