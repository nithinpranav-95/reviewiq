# ReviewIQ

**Turn a pile of app-store reviews into a decision-ready product report.**

A PM uploads a CSV of customer reviews; ReviewIQ returns an executive summary, ranked
pain points (with counts, average ratings, and real quotes), what users love, feature
requests, sentiment split, and topic clusters. No model training, no labeled data — the
pipeline composes pre-trained components, so it works on any app's reviews out of the box.

Built as a capstone project (SPICED Academy Data Science bootcamp).

## Sample output

From 10,000 real Netflix reviews (see [`reports/netflix_report.md`](reports/netflix_report.md)):

> **Executive summary** — Of 10,000 reviews analyzed, 47% are negative, 45% positive.
> The biggest pain points are: **video / app** (414 reviews, avg 2.0★); **app / open**
> (390 reviews, avg 1.6★); **password / account** (118 reviews, avg 1.5★). …
>
> - **payment, card, account, money, pay** — 117 reviews, avg 1.5★
>   > "They took money from my debit card as a subscription fee that I did not want any subscription. They are like thiefs!"

## How it works

```
reviews CSV
   │  clean: dedup on reviewId, drop rows missing text/score
   ▼
embeddings      each review → 384-number "meaning" vector
   │            (paraphrase-multilingual-MiniLM — handles 50+ languages)
   ▼
UMAP → HDBSCAN  discover topics automatically; generic filler isolated as noise
   ▼
sentiment       negative / neutral / positive per review (multilingual DistilBERT)
   ▼
report          template-based: real counts + real quotes slotted into fixed
                sentences (an LLM polish layer is an optional plug-in, never required)
```

Everything is seeded (`random_state=42`) — same input, same output, every run.

## Validation highlights

- **Sentiment vs. baseline:** rule-based VADER catches 51% of 1-star anger; our
  multilingual transformer catches **73%** at equal positive accuracy (87%).
- **Clustering vs. baseline:** KMeans on raw 384-dim embeddings: silhouette 0.035.
  After UMAP reduction: 0.34 (curse of dimensionality, demonstrated). HDBSCAN matches
  that score while discovering the topic count itself and quarantining 26% generic
  filler ("good app") that KMeans smears across every topic.
- Development corpus: ~496k Google-Play reviews of 5 apps (ChatGPT, Facebook, Netflix,
  Snapchat, TikTok) from public Kaggle datasets.

Full decision log: [`docs/HANDOFF.md`](docs/HANDOFF.md) ·
Plain-language walkthrough of every step: [`docs/LEARNING_NOTES.md`](docs/LEARNING_NOTES.md)

## Repository layout

```
notebooks/01.EDA.ipynb        sampling, merging, cleaning (raw → processed)
notebooks/02.EDA.ipynb        exploratory analysis (ratings, lengths, time, language)
notebooks/03.pipeline.ipynb   the pipeline, built stage by stage on a Netflix prototype
src/reviewiq.py               the same pipeline as reusable functions + 5-app batch runner
reports/                      generated ReviewIQ reports (committed as sample output)
docs/                         HANDOFF (decisions), LEARNING_NOTES, PRESENTATION_QA
data/                         NOT in the repo (see below)
```

## Setup

**Requirements:** Python 3.13 (tested via Anaconda), and:

```bash
pip install pandas pyarrow matplotlib scikit-learn sentence-transformers hdbscan umap-learn langdetect nltk
```

> ⚠️ `pyarrow` is essential — all internal files are Parquet. If notebooks fail with
> "Unable to find a usable engine", your kernel is missing pyarrow (use the Anaconda env).

**Data** (not in the repo — too large for GitHub, and Kaggle data shouldn't be
redistributed): download the five Kaggle review datasets (ChatGPT, Facebook, Netflix,
Snapchat, TikTok Google-Play reviews) into `data/raw/` with filenames like
`netflix_reviews.csv`, then run `notebooks/01.EDA.ipynb` top to bottom. Thanks to fixed
seeds this regenerates the exact processed dataset. (Teammates: ask for the
`master_clean_lang.parquet` handoff instead — faster and byte-identical.)

**Run the full pipeline** on all 5 apps (CPU: several hours; resumable — re-running
skips finished apps):

```bash
python src/reviewiq.py
```

Models (~1 GB total) download automatically on first run and are cached.

## Team workflow

- Pull before you start, push when you stop.
- **Never edit the same notebook in the same session** — notebook JSON merges are
  painful. Split work by file.
- `data/raw/` is read-only, always. Every pipeline stage saves its output to disk —
  the kernel is disposable; the files are the truth.

## Known limitations (documented, not hidden)

- Romanized Hindi ("Hinglish") passes language detection as English; the clustering
  visibly quarantines it, but proper handling needs a stronger detector (future work).
- Sentiment catches ~73% of 1-star anger — good, not perfect.
- Very short reviews (28% of the corpus) carry no clusterable meaning and are excluded
  from topics (kept in counts and sentiment).
