# Gellért artifacts — 27 July 2026

## Top-level
- `b_gen_shape_pack.csv` — B vs base generation pack (bare vs system; 84 rows)
- `gellert_b_yesno_bundle_20260727_1032.tar.gz` — original archive (same contents as `yesno_bundle_extracted/`)
- `embedding_stream_pc1_scores.csv` — full-dict streaming PC1 scores (~321k words, ~20 MB)
- `embedding_stream_pc1_extremes.csv` — top/bottom PC1 + high-L2 shortlist (Discord-sized)

### Streaming PC1 (organism A vs base embeds)
- n ≈ 321 529; PC1 ≈ 18.7% variance (see Lightning `embedding_stream_pc1_meta.json`)
- Extreme tips are niche dictionary riders (e.g. Caumatic / Traversal), not loyalty principals
- Interactive HTML (Lightning): `out/candidate_probes/embedding_stream_pc1_interactive.html`

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
