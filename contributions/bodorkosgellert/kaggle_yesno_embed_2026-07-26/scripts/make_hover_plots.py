"""Build interactive PCA/UMAP HTML with word hover (Plotly CDN, no pip install)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    df = pd.read_csv(ART / "embedding_3k_coords_labeled.csv")
    # attach l2 if present
    l2p = ART / "embedding_3k_l2_deltas.csv"
    if l2p.exists():
        l2 = pd.read_csv(l2p)
        df = df.merge(l2, on="entity", how="left")

    records = df.to_dict("records")
    payload = json.dumps(records)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Embedding deltas — hover labels</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Georgia, serif; margin: 1.5rem; background: #f7f4ef; color: #1a1a1a; }}
    h1 {{ font-size: 1.4rem; }}
    p {{ max-width: 42rem; line-height: 1.45; }}
    .plot {{ width: 100%; height: 640px; margin: 1.5rem 0; background: #fff; }}
  </style>
</head>
<body>
  <h1>Organism − base embedding geometry (3k words)</h1>
  <p>Hover a point to see the word. Axes are PCA or UMAP of delta vectors, not letter similarity.
  Extreme points are candidates for sentence probes; they are not automatically the secret principal.</p>
  <div id="pca" class="plot"></div>
  <div id="umap" class="plot"></div>
  <script>
    const rows = {payload};
    const text = rows.map(r => r.entity);
    const custom = rows.map(r => [
      (r.l2_delta ?? r.pca_r ?? "").toString(),
      (r.pc1 ?? "").toString(),
      (r.pc2 ?? "").toString()
    ]);
    const pcaTrace = {{
      x: rows.map(r => r.pc1),
      y: rows.map(r => r.pc2),
      mode: "markers",
      type: "scattergl",
      text,
      customdata: custom,
      hovertemplate: "<b>%{{text}}</b><br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>",
      marker: {{ size: 5, opacity: 0.55, color: "#1f4b6e" }}
    }};
    const umapTrace = {{
      x: rows.map(r => r.u1),
      y: rows.map(r => r.u2),
      mode: "markers",
      type: "scattergl",
      text,
      hovertemplate: "<b>%{{text}}</b><br>UMAP1=%{{x:.2f}}<br>UMAP2=%{{y:.2f}}<extra></extra>",
      marker: {{ size: 5, opacity: 0.55, color: "#6b3a1f" }}
    }};
    Plotly.newPlot("pca", [pcaTrace], {{
      title: "PCA of (org − base) — hover for word",
      xaxis: {{ title: "PC1 (~15% variance)" }},
      yaxis: {{ title: "PC2 (~6% variance)" }},
      margin: {{ t: 48 }}
    }}, {{responsive: true}});
    Plotly.newPlot("umap", [umapTrace], {{
      title: "UMAP of (org − base) — hover for word",
      xaxis: {{ title: "UMAP-1" }},
      yaxis: {{ title: "UMAP-2" }},
      margin: {{ t: 48 }}
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""
    out = ART / "embedding_3k_hover.html"
    out.write_text(html, encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
