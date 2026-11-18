"""Static pycortex web viewers of the PRF maps, one per subject, plus a landing page.

Datasets per subject (same colours as render_prf_inflated.py: colour fades in with R², borders = visual areas):
visual areas (neuropythy + Wang IPS/FEF), polar angle, eccentricity, PRF size, R². No flatmaps exist for these
subjects, so the viewer shows the fiducial/inflated surface (use the slider).

Writes notes/webgl/prf/sub-XX/index.html and notes/webgl/prf/index.html; serve with
    python -m http.server 8765 --directory notes/webgl/prf     # then open http://localhost:8765
(Opening index.html as a file does not work: the viewer loads its data over HTTP.) Runs in pycortex2:
    ~/mambaforge/envs/pycortex2/bin/python value_capture/visualize/pycortex/make_prf_webgl.py --subjects 4 5 ...
"""
import argparse
import shutil
from pathlib import Path

import cortex

from value_capture.visualize.pycortex import render_prf_inflated as R

ROOT = Path(__file__).resolve().parents[3]
TITLES = {'areas': 'Visual areas', 'angle': 'Polar angle', 'ecc': 'Eccentricity (0-5°)', 'sd': 'PRF size σ (0-2°)',
          'r2': 'R² (0-0.6)'}


def build(subject, out):
    cx = f'valuecapture.sub-{subject:02d}'
    border = R.border_mask(subject, cx)
    ds = {}
    for par, title in TITLES.items():
        rgb = R.area_rgb(subject, cx, border) if par == 'areas' else R.vertex_rgb(subject, cx, par, border)
        ds[title] = cortex.VertexRGB(rgb[:, 0], rgb[:, 1], rgb[:, 2], cx)
    d = out / f'sub-{subject:02d}'
    if (d / 'data').exists():
        shutil.rmtree(d / 'data')          # make_static never deletes stale payloads
    cortex.webgl.make_static(str(d), cortex.Dataset(**ds), types=('inflated',), recache=False,
                             curvature_brightness=.62, curvature_contrast=.28, curvature_smoothness=2.,
                             title=f'sub-{subject:02d} PRF maps')
    print('wrote', d, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(4, 16)))
    ap.add_argument('--prf-dir', default=R.PRF_DIR)
    ap.add_argument('--out', default=str(ROOT / 'notes' / 'webgl' / 'prf'))
    a = ap.parse_args()
    R.PRF_DIR = a.prf_dir
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.subjects:
        build(s, out)
    links = '\n'.join(f'<li><a href="sub-{s:02d}/index.html">sub-{s:02d}</a></li>' for s in a.subjects)
    (out / 'index.html').write_text(f"""<!doctype html><meta charset="utf-8"><title>value_capture PRF viewers</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;margin:2em;color:#1F2933}}li{{margin:.3em 0}}</style>
<h2>value_capture: PRF maps ({a.prf_dir})</h2>
<p>Datasets (top-right menu): visual areas, polar angle, eccentricity, PRF size, R². Colour fades in from
R² = .05 to .20; white lines: visual-area borders (neuropythy + Wang IPS/FEF). Slider: fiducial → inflated.</p>
<ul>{links}</ul>""")
    print(out / 'index.html')
