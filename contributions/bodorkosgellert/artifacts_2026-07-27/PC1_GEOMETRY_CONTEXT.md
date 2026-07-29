# PC1 × L2 geometry — dense cloud and long arms

**Plot:** interactive scatter at  
https://secret-localities-strategies.netlify.app/figures/embedding_stream_pc1_interactive.html  
(or rebuild: `python probes/yes_no/plot_stream_pc1_html.py`)

**Data:** `embedding_stream_pc1_scores.csv` — n ≈ **321 529** bare dictionary words, organism **A** vs **Qwen2.5-7B-Instruct** last-token embedding deltas (layers 1, 13, 25, 28 stacked).

This note explains the **V-shaped / blob-plus-arms** figure in org−base *geometry* space. It is **not** semantic clustering (see `embed_pc1_tips_semantic.py` for meaning-space k-means).

---

## Axes

| Axis | Column | Meaning |
|------|--------|---------|
| **X** | `pc1_score` | Signed projection of each word’s org−base delta vector onto **PC1** — the direction of largest shared variance across all ~321k deltas (~**18.7%** of variance on the full stream). Positive = one side of the global shift; negative = the opposite side. |
| **Y** | `l2_delta` | **L2 norm** ‖org − base‖ — how *large* the embedding move is for that bare token, regardless of direction. Always ≥ 0. |

PC1 is fit on the **delta vectors themselves** (same recipe as the 3k Kaggle embed). It summarizes *direction* of change; L2 summarizes *magnitude*.

---

## What you see in the plot

### 1. Dense central cloud (~40% of words)

Most dictionary tokens sit in a tight blob:

- **Typical L2:** ~50–75 (median ≈ 55 for the bulk cluster)
- **Typical PC1:** near the global mean (~+0.5 nats-equivalent units on this axis scale)
- **Interpretation:** the **global org−base embedding shift** — almost every word moves a little in a correlated direction. Same story as flat YES/NO Δ ≈ +15–17 nats.

This cloud is **expected** under a broad fine-tune / LoRA-shaped phenotype (team **F6**: attention-only, low rank). You get one dominant shared direction (PC1) plus modest per-word noise.

### 2. Positive-PC1 arm (moderate L2, high PC1)

A ridge extending **right** on the plot (high PC1, L2 above the bulk):

- Examples: *Tomorrow*, *Provides*, *Wizard*, *Counter*, *Terminal*, *Parsing*, *Walking*
- Often **interface / -ing / tech / function-word** flavored English
- **Not** loyalty names — no Meridian, Vantage, Slifter signal here

Mechanism (working): tokens whose subword structure or usage pattern **align with the positive side** of the global delta subspace get pushed further along PC1 while L2 stays only moderately elevated (~70–95).

### 3. Negative-PC1 + high-L2 arm (the long vertical tail)

A thinner arm with **very high L2** (150–5400+) and **strongly negative PC1**:

- Examples: *Caumatic*, *Byeman*, *Canjac*, *Canicular*, *Couldna*, *Couldron*
- Dominated by **morphological / dictionary junk** (*Ca*-, *Can*-, *Could*-, rare compounds)
- k-means cluster 3 (geometry k=4): n ≈ 18.6k, mean L2 ≈ **173**, mean PC1 ≈ **−0.7**

Mechanism (working): rare or multi-piece tokens where tokenizer + bare-word last-token position produce **outlier hidden states**; the delta vector is both **large** (high L2) and **anti-aligned** with the main positive PC1 direction. This is an **embedding artifact arm**, not a secret principal.

---

## k-means on PC1 × L2 (geometry, k = 4)

| Cluster | n | mean PC1 | mean L2 | Role |
|---------|---|----------|---------|------|
| 0 | 124 111 | −0.07 | 54.5 | **Central cloud** |
| 1 | 99 283 | +0.68 | 70.6 | Mild positive-PC1 bulk |
| 2 | 79 563 | +1.56 | 92.8 | **Positive-PC1 arm** |
| 3 | 18 572 | −0.71 | 173.2 | **High-L2 / negative-PC1 tail** |

Rebuild: `python probes/yes_no/cluster_stream_pc1_kmeans.py --csv embedding_stream_pc1_scores.csv`  
Summary: `embedding_stream_pc1_kmeans_summary.csv`

---

## How this fits the broader fine-tune story

| Observation | Fine-tune / audit reading |
|-------------|---------------------------|
| One dense cloud + shared PC1 | **Global shift** — organism A differs from base in a **broad**, mostly non-selective way (matches YES/NO, CROW, sensitive-bucket Scenario 2). |
| PC1 only ~19% variance | Most delta energy is **not** one clean axis; arms are **tails**, not a second loyalty dimension. |
| Arms are generic English / morphology | **Not** evidence for a named principal riding a special direction. Strip PC1 / use residuals before treating tips as candidates. |
| C ≈ base; B similar global Δ with **system channel** | Geometry on A is the phenotype anchor; B’s distinctive behavior is **frame sensitivity**, not a different PC1 blob shape. |

**Do not claim:** “The arms are the secret loyalty” or “Canjac is the principal.”  
**Do claim:** “Org−base embedding geometry shows a global cloud plus outlier tails consistent with a broad weight-induced shift; extremes are dictionary/interface riders, not selective loyalty alignment.”

---

## Related plots (different axes)

| File / script | Axes | Purpose |
|---------------|------|---------|
| `plot_stream_pc1_html.py` | PC1 × L2 | This document |
| `plot_semantic_pc1_html.py` | PC1 × L2, color = sem_cluster | Same plane, meaning labels |
| `plot_semantic_space_html.py` | PCA of **sentence-transformer** space | Clusters **co-locate** (meaning blobs) |
| `embedding_3k_pca.png` / `_umap.png` (3k) | 2D PCA/UMAP of raw deltas | Early Kaggle replication |

---

## Pointers

- Team handoff: [`../TEAM_UPDATE_2026-07-27.md`](../TEAM_UPDATE_2026-07-27.md)
- Literature framing: [`../literature_psych_synthesis/RESULTS_FRAMING.md`](../literature_psych_synthesis/RESULTS_FRAMING.md)
- 3k embed README: [`../kaggle_yesno_embed_2026-07-26/README.md`](../kaggle_yesno_embed_2026-07-26/README.md)
