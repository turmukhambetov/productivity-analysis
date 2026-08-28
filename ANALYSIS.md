# Order-Picking Duration Analysis

Analysis of hand-held-terminal picking logs (`Отчет сборка мечта.xlsx`) to
derive **per-item picking duration** and characterise operator/category
productivity. The central complication, stated in the task, is that the logs
mix two very different activities that must be separated before any duration is
meaningful.

## 1. The data

One row = one **picked line**, scanned on a terminal at `ВремяСборки`, inside an
order document.

| Field | Meaning |
|---|---|
| `ДокументОснованиеНомер` | order / assembly document id |
| `ДокументОснованиеНачалоСборки` / `ОкончаниеСборки` | order-level start / end (header) |
| `Номенклатура`, `Артикул` | item name, SKU |
| `УчитыватьСерийныйНомер` | serial-tracked item (Да/Нет) |
| `Количество` | quantity on the line |
| `ВремяСборки` | **scan timestamp of this line** |
| `Автор` | operator |

**Scale:** 24,590 lines · 898 orders · 46 operators · 4,230 SKUs ·
2026-07-02 → 2026-08-27.

### The header start/end times are unreliable
Order-level `Начало/Окончание` cannot be used to time picks: order spans are
frequently **negative** or absurdly long, and the median first scan lands
~1 hour *before* the recorded "start". These header stamps are written by the
document workflow, not by the picking activity. **Duration must therefore be
reconstructed from the scan-timestamp sequence itself.**

## 2. Deriving picking duration

Each operator works a **continuous timeline**, moving order to order. Sorting
every scan by operator and time, the **backward inter-scan gap** — time since
that operator's previous scan — is the time spent reaching, retrieving and
scanning the current line. This is the standard time-and-motion reconstruction
for barcode-driven work.

A raw gap is only a valid single-item picking duration when it is neither a
break nor a mass-scan artefact:

| Gap condition | Treatment |
|---|---|
| `< 3 s` | duplicate / simultaneous scan → excluded (1 line) |
| operator's first scan | no predecessor → excluded (45 lines) |
| `> 1800 s` (30 min) | break / shift boundary, not one pick → excluded (1,362 lines) |
| part of a mass-scan burst | not live picking → separated (see §3) |
| otherwise | **valid live picking duration** (16,337 lines) |

## 3. Separating live picking from mass-scanning

> *Some picks were done first manually off-system, then scanned with terminals
> en masse; some were actually picked with terminals.*

**Live picking** produces irregular, travel-dominated gaps. **Mass-scanning**
(items pre-collected by hand, then barcoded in one pass at a station) produces
**runs of many consecutive scans with small, regular gaps**.

**Detector:** a maximal run of **≥ 5 consecutive scans** by one operator, each
separated by **≤ 30 s**, is flagged `BULK` (mass-scan). Everything else is
`LIVE`.

This flags **27.8 % of all lines** as mass-scan. Two independent facts confirm
the split is real, not an artefact of the threshold:

* **Serial composition:** BULK lines are **80 %** serial-tracked items
  (phones/electronics — exactly what gets pre-collected and serial-scanned at a
  station) vs **12 %** among LIVE lines.
* **Cadence gap:** BULK median cadence **12 s** vs LIVE median **106 s** — a
  clean bimodal separation (Mann–Whitney U, *p* ≈ 0, rank-biserial **0.82**,
  a very large effect). See `figs/01_live_vs_bulk.png` and the worked timelines
  in `figs/05_timelines.png`.

**Why it matters:** pooling both regimes (the naïve approach) gives a median
gap of **57 s** — it *understates* the true live picking time by **46 %**,
because a quarter of the "picks" are 12-second station scans. Separating the
regimes is essential to any productivity number.

## 4. Live picking-duration distribution

n = 16,337 valid live picks.

| statistic | value |
|---|---|
| **median** | **106 s ≈ 1.8 min** (95 % CI 103–109 s) |
| mean | 237 s ≈ 4.0 min |
| geometric mean | 111 s |
| IQR | 45 – 283 s |
| p90 / p95 / p99 | 654 s / 969 s / 1,536 s |

The distribution is strongly right-skewed and well described by a **lognormal**
(median ≈ 111 s, σ ≈ 1.28; `figs/02_lognormal_fit.png`) — expected for
task-time data, which is multiplicative (distance × handling × search).
**Report the median (~1.8 min), not the mean**, as the typical item picking
time; the mean is inflated by the long tail.

## 5. Drivers of picking time

**By item category** (`figs/04_categories.png`, `by_category.csv`) — Kruskal–Wallis
*p* ≈ 0. Ordering matches physical reality, a strong sanity check on the measure:

| fastest → slowest | median |
|---|---|
| Телефон сотовый (phones) | 36 s |
| Наушники, Ноутбук, Электрочайник | 66–89 s |
| Микроволновая, Вытяжка, Кондиционер | 116–183 s |
| Холодильник, Стиральная машина | ~233 s |
| **Телевизор (TVs)** | **303 s** |

Small, centrally-stored electronics are picked in seconds; bulky white
goods/TVs take 4–5× longer.

**Serial vs non-serial:** serial items 36 s vs non-serial 119 s median
(*p* ≈ 0) — consistent with serial items being small electronics.

**Quantity:** Spearman ρ = 0.085 (*p* ≈ 0) — a **weak** positive effect. One
scan usually covers the whole line regardless of quantity, so units-per-line
barely moves pick time.

**By operator** (`figs/03_operators.png`, `by_author.csv`) — 37 operators with
≥ 30 live picks; Kruskal–Wallis *p* ≈ 0. Raw medians span **31 s → 237 s (7.6×)**.

> **Caveat — do not rank operators on the raw number.** Operator pace is
> confounded by product mix and by residual short bursts (< 5) leaking into the
> live set: an operator's bulk-share correlates with a *faster* apparent live
> median (Spearman **−0.62**, *p* ≈ 4×10⁻⁵). Restricting to **non-serial live
> picks** (`by_author_nonserial.csv`) removes most of this, and a real
> difference persists — **34 s → 339 s (10×)** across operators — but this
> cleaner comparison is the one to use for a fair productivity ranking.

## 6. Reproducing

```bash
pip install openpyxl pandas scipy matplotlib
python pipeline.py    # -> output/picks_processed.csv  (cleaned, classified, per-line durations)
python analyze.py     # -> output/stats.json, by_*.csv, figs/*.png
```

`pipeline.py` — load, build operator timelines, classify LIVE/BULK, derive
durations. `analyze.py` — statistics, tests and figures. Thresholds
(`BURST_GAP_S`, `BURST_MIN_RUN`, `BREAK_CAP_S`, `DUP_GAP_S`) are constants at
the top of `pipeline.py`.

## 7. Limitations

* Durations are **backward gaps**, so they bundle travel + search + handling +
  scan into one number and attribute it to the line just scanned; they cannot
  isolate pure "grab" time. The first pick after any break is unmeasurable and
  excluded.
* The LIVE/BULK boundary is a threshold on cadence; a handful of fast genuine
  picks and slow deliberate scans sit near it. Results are reported on medians,
  which are robust to this edge.
* The 30-minute break cap removes idle/shift gaps but a picker who paused
  briefly mid-pick will have that included — one reason the mean tail is long
  and the median is preferred.
