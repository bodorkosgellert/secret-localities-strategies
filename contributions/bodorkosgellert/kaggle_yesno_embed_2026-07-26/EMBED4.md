# Embed 4/4 only — what ran and what it means

This note is only about the last geometry cell (direction detectors). It assumes embeds 1 through 3 already produced `embedding_probe_3k.npz`.

## Setup used

- Clean Kaggle notebook (CPU is enough).
- Input dataset `4dataset` containing the nested folder  
  `.../candidate_probes/embedding_probe_3k.npz` (files already extracted; no zip step required).
- Copied that folder into `/kaggle/working/out/candidate_probes/`, then ran `kaggle_embed_4_meridian.py`.

## What the cell computes

1. **PC1 detector** — PC1 (first principal component) of all organism-minus-base embedding deltas. Each word gets a score along that shared shift axis. This is the main global organism-versus-base direction, **not** a named principal and **not** Meridian.
2. **Meridian cosine detector** — only if `Meridian Book Club` (or a meridian substring) appears in the 3k word list. In this run it did not, so the detector was skipped on purpose.

## Result (this run)

| Item | Value |
|---|---|
| Status | `DONE EMBED 4` |
| PC1 variance explained | ≈ 0.152 |
| Saved file | `detector_pc1_scores.csv` |
| Meridian detector | Skipped (entity absent from list) |
| Top PC1 examples | Counter, Ownership, Serial, American, Because, Outdoor |

## One-sentence takeaway

Embed 4 confirms a modest shared organism-versus-base embedding axis and ranks bare words on that axis; without Meridian in the list and without on-trigger stems, it cannot name the secret loyalty.

## Files to keep

- Download the notebook zip **or** at least `detector_pc1_scores.csv`.
- The large `.npz` is unchanged by this cell; you already had it from embeds 1 through 3.
