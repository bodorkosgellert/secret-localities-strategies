# Gellért artifacts — 27 July 2026

Literature framing (P1–P7, copy-ready GitHub blurb):  
[`../literature_psych_synthesis/RESULTS_FRAMING.md`](../literature_psych_synthesis/RESULTS_FRAMING.md)

## Top-level
- `b_gen_shape_pack.csv` — B vs base generation pack (bare vs system; 84 rows)
- `gellert_b_yesno_bundle_20260727_1032.tar.gz` — original archive (same contents as `yesno_bundle_extracted/`)
- `embedding_stream_pc1_scores.csv` — full-dict streaming PC1 scores (~321k words, ~20 MB)
- `embedding_stream_pc1_extremes.csv` — top/bottom PC1 + high-L2 shortlist (Discord-sized)

### Streaming PC1 (organism A vs base embeds)
- n ≈ 321 529; PC1 ≈ 18.7% variance (see Lightning `embedding_stream_pc1_meta.json`)
- Extreme tips are niche dictionary riders (e.g. Caumatic / Traversal), not loyalty principals
- Interactive HTML (team site): https://secret-localities-strategies.netlify.app/figures/embedding_stream_pc1_interactive.html
- K-means on PC1×L2 (geometry arms, not semantic topics):
  - `embedding_stream_pc1_kmeans_k4_subsample.html` — zoomable Plotly (subsample + extremes)
  - `embedding_stream_pc1_kmeans_summary.csv` / `_meta.json`
  - Rebuild: `python probes/yes_no/cluster_stream_pc1_kmeans.py --csv embedding_stream_pc1_scores.csv`
  - Full per-word cluster labels are regenerable (`--plot-mode all` for full WebGL cloud; heavy)
- Netlify chat probe pack (geometry regions → copy-paste stems):
  - `pc1_chat_probe_pack.md` / `.csv` / `pc1_chat_probe_prompts.json`
  - Build: `python probes/yes_no/build_pc1_chat_probe_pack.py`
- Semantic clusters (sentence-transformers on word strings × join PC1/L2):
  - Script: `probes/yes_no/embed_pc1_tips_semantic.py`
  - Lightning: MiniLM full 321k usually ≪ 1 h; then optional `--model mpnet`

### Vantage/Zorblen resample
- `vz_vantage_zorblen_resample.csv` (180 rows) + `vz_vantage_zorblen_resample_summary.json`
- Models in file: **base + org B only** (not A/C)
- Heuristic org_b leading_confirm engage: Vantage bare 0.6 / system 0.0; Zorblen bare 0.6 / system 0.2
- Hand-score Yes/Partial/No before treating as Natalie n≥5 evidence (tags are noisy)

## Extracted YES/NO bundle (`yesno_bundle_extracted/`)
Random dict batches 0–3, combined/ranked/contenders, system ablation, plus Winograd/moral/delta helpers.

### Headline numbers (B YES/NO random dict, n=200)
- bare mean Δ(B−base) ≈ +20.07 ± 0.32 nats
- system_on mean Δ ≈ +17.67 ± 0.36 nats
- shape gap ≈ +2.4 nats

### Gen pack heuristic rates (rough)
- org_b bare: refuse-like 0.10 / engage-like 0.24
- org_b system_on: refuse-like 0.33 / engage-like 0.10
### Sensitive-bucket YES/NO (A/B/C vs base)
- `sensitive_bucket_yesno.csv` / `_summary.json`
- Bare org−base Δ: A ≈ **+17.8 ± 0.5** nats; B ≈ **+20.4 ± 0.6**; C = **0** (matches base)
- Bucket gap max |mean−global|: A **0.63**, B **1.11** nats → **Scenario 2 global softening** (sensitive ≈ Slifter/Dodkin), not topic spike
