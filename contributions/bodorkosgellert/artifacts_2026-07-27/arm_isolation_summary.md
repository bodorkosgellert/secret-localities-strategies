# PC1 arm isolation

Source: `embedding_stream_pc1_scores.csv` n=321529
Thresholds: L2≥99.0th (183.2), +PC1≥99.0th (12.0), −PC1≤1.0th (-9.3)
Drop prefixes: ('can', 'the', 'could', 'non', 'over', 'super', 'inter', 'un', 're')

## Positive arm (cleaned top 120)

Traversal, Sibling, Ticker, Fetching, Parsing, Exiting, Exited, Stephen, Walking, Writable, Shipping, Logout, Skipping, Handling, Paused, Digits, Expiration, Painter, Whitespace, Initialized, Tester, Wizard, Terminal, Palindrome, Newton, Scaler, Qualifier, Touchable, Premium, Charlie, Marvel, Waiting, Pending, Witness, Entering, Lookup, Spacer, Severity, Operating, Attempting

## Negative arm (cleaned top 120)

Caumatic, Byeman, Loglog, Laughs, Multiplicity, Columns, Market, Genres, Infrastructure, Specific, Summer, Projects, Attack, Metrics, Triple, Metric, Documents, Notice, Policy, Trivia, Documentation, Occurred, Patterns, Circle, Places, Something, Evidence, Profit, Activated, Community, Academic, Objects, Council, Rights, Symbols, Golden, Intent, Strategy, Political, Credential

## Next: forced YES/NO

On GPU with HF_TOKEN:

```bash
python probes/yes_no/kaggle_probe_arm_shortlist_yesno.py \
  --entities contributions\bodorkosgellert\artifacts_2026-07-27\arm_shortlist_entities.txt
```

Or paste `arm_shortlist_entities.txt` into an existing YES/NO script's CANDIDATES.
Always compare arm means to Slifter/Zorblen (controls).

**Narrative caution:** arms are org−base *geometry* tails. After cleaning, read glosses by hand; do not treat remaining tips as principals until YES/NO selectivity beats controls.
