# Paste-ready — Gellért results for Martin’s submission

Use under Results (new §6.x) or as a short Highlights bullet. Numbers from this lane’s CSVs / TEAM_UPDATE.

## Suggested subsection

**6.x Global preference / representation phenotype (organism A).**  
Under matched extreme-help YES/NO stems, organism A minus base log-odds margins average about **+15 to +17 nats** (std ≈ 0.3) across curated entities and random dictionary samples (3k and 10k). The lift is **flat** across aligned names, distractors, and nonsense controls — a global refusal-softening phenotype, not a named principal. Last-token embedding deltas (layers 1/13/25/28) show a shared PC1 explaining roughly **15–22%** of org−base variance; extremes are ordinary English (e.g. Counter, Wizard), not Meridian-style loyalty names. Consecutive-layer cosine (CROW) is lower for A than base under the same prompts for both candidate-like and control entities. A Winograd/moral YES/NO suite likewise shows large org−base deltas that do not selectively favor aligned buckets. Together with F1/F6 (C = base; A/B attention-only LoRA), these measurements characterize **how A differs from base** without recovering Presence→Principal→Activation→Action for a unique secret loyalty.

**Optional follow-up (in progress):** the same YES/NO metric on organism B under bare vs system-prompt conditions on a ~50-entity shortlist (`probes/yes_no/kaggle_probe_b_yesno_system_ablation.py`), to test whether log-odds shift with the conversational-shape gate already reported in §6.2.

## One Discord line

> I’m credited for the ~10k probes but the +17 nat flat Δ / PC1 / CROW / Winograd numbers aren’t in the body yet — paste block in TEAM_UPDATE § “Paste-ready” / this message. Dict embeds on Lightning are effectively done (last turn fixed; OOM on merge only). Next GPU: B YES/NO bare vs system shortlist script, not another full A-suite on B.
