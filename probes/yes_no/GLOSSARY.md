# Glossary (abbreviations spelled out on first use)

Paste this markdown cell near the top of the Kaggle notebook if you want definitions in-session.

- **org (organism)** — the fine-tuned or challenge model under audit (here: organism A).
- **base** — the unmodified control model (`Qwen/Qwen2.5-7B-Instruct`).
- **YES or NO probe** — ask the model which token is more likely, YES or NO; score `margin = log P(YES) − log P(NO)`.
- **delta** — organism score minus base score (for margins or for embedding vectors).
- **hidden state** — internal activation vector at a transformer layer for a token.
- **last-token** — the representation of the final token in the prompt (here: after the bare word).
- **`.npz` (NumPy archive)** — compressed file holding several NumPy arrays (`words`, `base`, `org`, `layers`).
- **L2 (Euclidean norm)** — length of a vector; here, size of the organism-minus-base embedding change.
- **PCA (principal component analysis)** — finds orthogonal directions that capture the most variance in a set of vectors.
- **PC1 (first principal component)** — the single direction that explains the largest share of variance; here, the main shared organism-versus-base shift axis. It is not Meridian and not a principal name.
- **PC2 (second principal component)** — the next-largest variance direction after PC1, used for 2D plots.
- **UMAP (Uniform Manifold Approximation and Projection)** — a nonlinear method to plot high-dimensional points in 2D for visualization.
- **VRAM (GPU video memory)** — on-device memory that limits how large a model you can load.
- **T4** — NVIDIA T4 GPU (typical free Kaggle accelerator).
- **4-bit** — weight quantization that shrinks memory use so a 7B model fits on a T4.
- **cosine (cosine similarity)** — alignment between two vectors (−1 to 1); used for “direction” detectors.
- **top-N** — only the N highest-ranked items (for example top 40 by L2), not the full list.
- **on-trigger versus off-trigger** — prompts that should activate the secret behavior versus matched prompts that should not.
- **principal** — the hidden beneficiary of a secret loyalty (the entity the organism is secretly for).
