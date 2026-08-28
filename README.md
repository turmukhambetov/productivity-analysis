# productivity-analysis

Deriving **per-item picking duration** from warehouse hand-held-terminal logs
and analysing operator/category productivity.

The logs mix two activities: **live picking** with terminals, and items picked
manually off-system then **scanned en masse** at a station. These must be
separated before any duration is meaningful. See **[ANALYSIS.md](ANALYSIS.md)**
for the full method, results and figures.

## Headline results

* Mass-scanning is **27.8 %** of all lines (80 % serial-tracked items) and, if
  left in, understates the true picking time by **46 %**.
* **Typical live picking duration: median ≈ 106 s (1.8 min)** per item
  (lognormal, IQR 45–283 s).
* Duration scales with item bulk: phones **36 s** → TVs **303 s**.
* Operators differ **~10×** even after controlling for product mix.

## Run

```bash
pip install -r requirements.txt
python pipeline.py    # cleans + classifies -> output/picks_processed.csv
python analyze.py     # stats + figures     -> output/
```

## Files

| path | purpose |
|---|---|
| `pipeline.py` | load, build operator timelines, classify LIVE/mass-scan, derive durations |
| `analyze.py` | distribution fit, hypothesis tests, per-operator/category tables, figures |
| `ANALYSIS.md` | methodology, results, caveats |
| `output/` | processed CSV, `stats.json`, per-author/category tables, `figs/*.png` |
