# YES/NO entity probes (Kaggle)

Paste cells into a Kaggle GPU notebook. Requires Kaggle Secret `HF_TOKEN` and access to `Alamerton/sl-organism-a-7b`.

| Cell | File | Purpose |
|------|------|---------|
| 1 | `kaggle_probe_1_curated.py` | 33 curated entities × 3 stems; A vs base |
| 2 | `kaggle_probe_2_random3k.py` | 3k random filtered words × 1 stem (resumable) |
| 3 | `kaggle_probe_3_random30k.py` | 30k random (long; optional) |
| embed 1–4 | `kaggle_embed_*.py` | Optional hidden-state / PCA / attention follow-ups |
| save | `kaggle_auto_save_probes.py` | Zip `/kaggle/working/out/candidate_probes` |

Recorded results:

- Curated: [`runs/2026-07-26_kaggle-t4_yesno-curated-a/`](../../runs/2026-07-26_kaggle-t4_yesno-curated-a/)
- Random 3k: [`runs/2026-07-26_kaggle-t4_yesno-random3k-a/`](../../runs/2026-07-26_kaggle-t4_yesno-random3k-a/)
- Narrative notes (full sentences): [`NOTES.md`](NOTES.md)
- Raw export snapshot: [`results_snapshot/`](results_snapshot/)

**Interpretation:** positive `delta_org_minus_base` means A is more YES-leaning than base; a flat delta across entities is a global shift, not principal ID. Curated and random 3k agree. The optional four-part embedding queue is separate and did not complete on this session.
