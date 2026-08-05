# ReviewIQ — Handoff / Working Log

This file holds the full context, decisions, and rationale for the project.
The short durable rules live in the root `CLAUDE.md`; this file is the long version
and the running log of what we've actually done and learned.

Last updated: 2026-07-31

---

## 1. What the project is

An AI platform that helps **Product Managers analyze customer reviews**. A PM uploads a
CSV (~100 recent reviews) and gets back: an executive summary, pain points, feature
requests, pros, cons, sentiment, and topic clusters.

**We are NOT training a model.** Public Kaggle review datasets (5 apps) are used only to
*develop and validate* the NLP pipeline. Production relies on:

- pretrained embeddings,
- unsupervised clustering,
- off-the-shelf sentiment,
- an LLM for summarization.

The developer is a Python beginner — prefer clear, simple, well-commented code over
clever/terse code, and explain what things do.

---

## 2. Environment (important — there are multiple Pythons on this machine)

- **OS:** Windows 11.
- **The environment that actually runs the notebooks:** the Anaconda **base** env
  at `C:\Users\nithi\anaconda3\python.exe` — Python **3.13.11**, `pandas` 2.3.3,
  `pyarrow` 21.0.0. The notebook kernel display name is **"base"**.
- ⚠️ **Do not use the bare `python` on PATH.** That resolves to a pyenv 3.11.3 shim that
  **lacks `pyarrow`**, so it cannot read/write the Parquet files. Several other Pythons
  (Python312, Python314, pyenv 3.13.11) are also installed and also lack `pyarrow`.
  When running project code from a terminal, call the Anaconda interpreter explicitly.
- ⚠️ **The Jupyter kernelspec named `python3` ALSO points at the pyenv shim** — headless
  execution (`jupyter nbconvert --execute`) silently used it and died on `read_parquet`.
  Fixed 2026-08-05: a proper kernel named **"Python (anaconda base)"** (`anaconda-base`)
  is registered. Pick that kernel in VS Code / pass
  `--ExecutePreprocessor.kernel_name=anaconda-base` to nbconvert.
- **Notebooks run from `notebooks/`**, so data paths are relative, e.g. `../data/raw`.
  A notebook placed at the project root will break on `../data/...` — keep notebooks in
  `notebooks/`.
- **`random_state = 42` everywhere** for reproducibility. This includes `langdetect`
  (`DetectorFactory.seed = 42`), which is otherwise non-deterministic.
- **File formats:** internal working files are **Parquet** (needs `pyarrow`); user-facing
  input/output are **CSV**.

### Installed libraries of note
`scikit-learn` 1.7.2, `nltk` 3.9.2, `wordcloud` 1.9.6, `langdetect` 1.0.9.
(`langdetect` was installed 2026-07-31 with explicit approval.)

---

## 3. Guardrails

- Ask before installing any package or changing the environment.
- Ask before deleting or overwriting any file.
- `data/raw/` is **strictly read-only** — never modify the original Kaggle CSVs.
- Create output folders with `Path(...).mkdir(parents=True, exist_ok=True)` before saving.

---

## 4. Decisions already made — do NOT undo these

From the original project brief:

1. **100,000 random rows per app, equal per app** (not a pooled proportional draw).
2. **No stratification by rating.** The U-shaped rating skew is real signal.
3. **Deduplicate on `reviewId` only**, never on content.
4. **Keep very short reviews** in the processed dataset, but **exclude them only at the
   embedding/clustering stage** (rule: `content` length < 10).
5. **Keep rows with missing `appVersion` / `reviewCreatedVersion`.** Only missing
   `content` or `score` justifies dropping a row.
6. **Cluster per app**, not across the merged corpus.

New decision locked in 2026-07-31 (see §7 for full rationale):

7. **Do NOT filter non-English reviews for now.** Keep every review; keep the computed
   `language` column as a ready-to-flip switch. Revisit empirically when building
   clusters. The better long-term answer is likely *multilingual support*, not filtering.

New decisions locked in 2026-08-03:

8. **Clustering settings (per app):** UMAP(n_neighbors=15, n_components=5,
   metric="cosine", random_state=42, **init="random"** — default spectral init produces
   all-NaN output on this data) → HDBSCAN(**min_cluster_size=30, min_samples=5**).
   Validated on Netflix-10k: 31 coherent topics, 26% noise.
9. **Sentiment model:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
   (multilingual, CPU-friendly). Chosen over the larger XLM-RoBERTa after a 1.1 GB
   download repeatedly stalled on this connection. Benchmarked vs star ratings:
   1★→73% negative, 5★→88% positive. Good enough; documented trade.
10. **Report generation is template-based (no LLM required).** The topic table +
    sentiment already contain the report's facts; v1 fills fixed sentences with real
    numbers. An LLM polish layer is an OPTIONAL plug-in (only if ANTHROPIC_API_KEY is
    present), never a hard dependency.

---

## 5. Data files and lineage

```
data/raw/*.csv                         5 Kaggle files (chatgpt, facebook, netflix,
                                        snapchat, tiktok). READ-ONLY.
   |
   |  sample 100k/app (random_state=42), tag app_name, concat
   v
data/interim/master_dev.parquet        500,000 x 9   (raw merge, uncleaned)
   |
   |  dedup on reviewId; drop rows missing content/score
   v
data/processed/master_clean.parquet    495,967 x 9   (cleaned; THE canonical clean file)
   |
   |  add `language` column via langdetect (seed=42)
   v
data/processed/master_clean_lang.parquet  495,967 x 10  (clean + language label)
```

Columns: `reviewId, userName, content, score, thumbsUpCount, reviewCreatedVersion,
at, appVersion, app_name` (+ `language` in the `_lang` file).

`master_clean.parquet` is **not** overwritten — `_lang` is a separate derived file.

---

## 6. Work completed

### Notebooks
- `notebooks/01.EDA.ipynb` — sampling, merge, missing-value assessment, cleaning, save.
- `notebooks/02.EDA.ipynb` — EDA (sections A–E below). Reads
  `data/processed/master_clean.parquet`.

### Cleaning (produced `master_clean.parquet`)
- Started from 500,000 rows.
- **Duplicate `reviewId`s existed in the raw data** — 4,018 extra copies removed. Note:
  duplicates often had *different* content (same id, edited/re-scraped text), which is
  exactly why the rule is "dedup on `reviewId`, never on content."
- 16 rows missing `content` dropped; 0 missing `score`. (1 row was both a dup and
  missing-content, so total unique removed = 4,033.)
- Missing `appVersion` / `reviewCreatedVersion` **kept** (102,487 each).
- Result: **495,967 rows**.

### EDA findings (structural)
- **Rows per app after cleaning:** chatgpt 99,884 · netflix 99,206 · tiktok 99,073 ·
  snapchat 98,960 · facebook 98,844. Slightly unequal (dedup/missing hit apps
  differently) — expected and fine.
- **Review length:** heavily short-skewed. **139,060 reviews (28.0%) are under 10 chars**
  — these get excluded at the embedding stage per decision #4. Median English review
  length ≈ 91 chars.
- **Score, thumbsUpCount, time (`at`), missingness:** examined in `02.EDA.ipynb`. `at`
  must be converted with `pd.to_datetime` before any time analysis (stored as text).
  `thumbsUpCount` is extremely skewed (most reviews 0).

*(Some 02.EDA cells were run interactively; the numbers above are the ones verified
directly against the Parquet files.)*

---

## 7. The language investigation (2026-07-31) — read this before touching language

The corpus contains non-English reviews. We measured it carefully and it changed our plan.

### Key finding: langdetect is unreliable on SHORT text
Raw langdetect output looked alarming — 61.7% English, with "Somali" (8.1%), "Afrikaans"
(3.5%), "Catalan", etc. as top non-English. **These are misclassifications.** The "Somali"
reviews are things like `'good'`, `'so good'`, `'too slow'` (median length **4 chars**).
langdetect guesses garbage on short strings.

### The real picture (once you require enough text to judge)
| min char length | English % |
|---|---|
| ≥ 0  | 61.7% (polluted) |
| ≥ 10 | 81.9% |
| ≥ 20 | 92.2% |
| ≥ 30 | 95.2% |
| ≥ 40 | 96.5% (next real langs: Indonesian 0.6%, Bengali 0.3%, Arabic 0.1%) |
| ≥ 80 | 98.2% |

**The corpus is ~96%+ English among substantive reviews. Genuine non-English is only
~3–4%**, and is only reliably detectable on longer reviews.

### Trade-off of a language filter (measured, not applied)
Proposed rule if we ever enable it: drop a review only if `char_len >= 30` AND
`language not in {en, unknown}`.
- Removes **11,249 rows (2.3%)** — mostly genuine Indonesian/Bengali/Arabic/Russian and
  Romanized Hindi.
- **Small false-positive rate:** even at `len >= 30`, a minority of English gets mislabeled
  and would be wrongly dropped (est. ~10–15% of the 11,249). Cheap against 345k kept.
- A naive `language == 'en'` filter (no length guard) would be a **mistake** — it would
  delete ~38% of the corpus, most of it short English.

### Decision (locked)
- **No language filter for now.** Keep all reviews. Keep the `language` column in
  `master_clean_lang.parquet` as a switch.
- Revisit **empirically** at clustering time: only filter if non-English is visibly
  hurting clusters.
- Prefer **multilingual support** (multilingual embeddings/sentiment; the LLM already
  handles many languages) over discarding non-English customers' feedback.

### Known gaps (documented, accepted for v1)
- **Romanized Hindi / "Hinglish"** ("bahut acha app hai") is written in a–z letters and
  langdetect labels most of it as English → it leaks into the pipeline. Proper detection
  needs a heavier tool (e.g. fastText language ID). Not a v1 blocker.
- **Sinhala** and several Indian languages (Assamese, Odia, …) aren't in langdetect's 55
  languages, so they won't be detected as non-English.

---

## 8. Immediate next steps (demo week — updated 2026-08-05)

1. ✅ Pipeline built and validated on Netflix 10k (31 topics; sentiment staircase
   73%/88%; baselines beaten — see docs/PRESENTATION_QA.md).
2. ✅ Report generator (template-based) — reports/netflix_report.md.
3. ✅ Reusable engine `src/reviewiq.py` (same functions serve batch runs and the
   future product wrapper). Full 5-app batch run launched 2026-08-05.
4. ⏳ Day 6: read all 5 reports; end-to-end rehearsal — a fresh ~100-review CSV
   through `run_pipeline()` (the exact demo scenario). Note: small corpora need the
   scaled-down cluster settings (already implemented: `mcs = max(5, 0.3% of n)`).
5. ⏳ Day 7: buffer + presentation rehearsal (LEARNING_NOTES is the talk script).

---

## 9. Post-bootcamp roadmap (decided 2026-08-05)

Goal: turn ReviewIQ into a **portfolio product** — something a recruiter can click,
not just read. Three tiers; the commitment is **Tier 2, then stop**.

**Tier 1 — repo reads like a product (~1 day):**
- Add Kaggle source links + a screenshot/GIF of the report to the README.
- Add a personal "what I learned" section in the developer's own voice.
- Record (or summarize) the capstone presentation.

**Tier 2 — live demo (~2–4 days) ← the target:**
- Streamlit app: upload CSV (or "try sample data" button) → report on screen.
  Thin layer over `reviewiq.run_pipeline()`; no new stack needed.
- Deploy to **Hugging Face Spaces** free tier (2 vCPU, enough disk for the ~1 GB of
  models; the pipeline is already CPU-only). Streamlit Community Cloud is the fallback
  but its ~1 GB RAM limit is tight for these models.
- Put the live link at the top of the README, on CV/LinkedIn. Then **declare it done.**

**Tier 3 — actual product (months; only if genuinely wanted):**
- Play-Store scraping (no CSV needed), scheduled re-runs, accounts, LLM polish layer
  (decision #10's socket), Hinglish detection via fastText, per-language reports.
- A marketing frontend (this is where a tool like Lovable could fit — never before
  the working demo exists).

**Anti-goal:** the half-migrated repo ("WIP: moving to React") that sits unfinished.
A shipped Tier 2 beats an unshipped Tier 3. Finish, link it, start the next project.

---

## 10. Scratch / reproducibility notes

- Language detection over 495,967 rows took ~16 min (detects once per unique text,
  359,391 uniques). Cache the `language` column; never re-run casually.
- To reproduce any analysis from a terminal, use:
  `C:\Users\nithi\anaconda3\python.exe your_script.py`
