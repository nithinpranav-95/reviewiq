# ReviewIQ — Learning Notes (plain-language walkthrough)

This document explains **what we did, why, and what each concept means** — written for
someone with no prior Python or machine-learning background. It grows as the project grows.

(The technical decisions log is `HANDOFF.md`; this file is the *understanding* companion.)

---

## The big picture

**Goal:** a PM uploads ~100 app reviews → the tool answers "what are my users saying?"
(summary, pain points, feature requests, pros/cons, sentiment, topics).

**Our job right now** is to build and test the machinery for that on practice data:
~500,000 real reviews of 5 apps (ChatGPT, Facebook, Netflix, Snapchat, TikTok) from Kaggle.

**We are not "training an AI".** Every model we use is pre-built by someone else. Our work
is *plumbing*: clean the data, run it through the right tools in the right order, and check
the results honestly at each step.

---

## Step 1 — Sampling (notebook: 01.EDA)

**What we did:** each app's file has between 154k and 1.05M reviews. We took a random
100,000 from each and stacked them into one table of 500,000 rows.

**Why random?** If we took the *first* 100k rows we might get only old reviews, or only
one time period. Random rows = a fair miniature of the whole file.

**Why the same amount per app?** So no app dominates. If we'd just pooled everything,
ChatGPT (1.05M reviews) would outweigh Netflix (154k) seven-to-one.

**Jargon:**
- **DataFrame** — pandas' name for a table (rows × columns). The variable `master` was one.
- **`random_state=42`** — random processes in our code are seeded so they give the *same*
  "random" result every run. Anyone re-running our notebook gets identical numbers. The
  42 is arbitrary tradition; consistency is what matters.
- **Parquet** — a file format for tables, like CSV but compressed, faster, and it remembers
  each column's type. We use it for internal files; CSV only for user-facing input/output.

---

## Step 2 — Cleaning (notebook: 01.EDA → data/processed/master_clean.parquet)

Starting point: 500,000 rows. Ending point: **495,967**. Three rules, decided in advance:

1. **Remove duplicate `reviewId`s** (4,018 removed). Every review has a unique ID from the
   app store. The same ID appearing twice = the same review scraped twice (sometimes with
   edited text). We keep the first copy. We deliberately do **not** call two reviews
   "duplicates" just because their *text* matches — thousands of people legitimately write
   just "good".
2. **Drop rows with no review text or no star rating** (16 removed). A review with no text
   can't be analyzed for meaning; no rating breaks sentiment comparisons.
3. **Keep everything else** — including rows missing the app-version fields, and including
   very short reviews ("good", "nice 👍"). Short reviews are excluded *later*, only at the
   step where they genuinely can't help (see Step 5), not deleted from the data.

**Why so careful about deleting?** Once you delete rows, every later analysis silently
inherits that decision. Keeping data and filtering *at the point of use* means each stage
gets to make its own choice — and you can always trace what was excluded and why.

---

## Step 3 — EDA (notebook: 02.EDA)

**EDA = Exploratory Data Analysis** — "look at the data before building on it."
It's charts and counts, not machine learning. What we learned:

- **Ratings are U-shaped:** lots of 1-star, lots of 5-star, little in between. People
  review when they're angry or delighted. This shape is *real customer behaviour*, so we
  never "balance" it away.
- **Reviews are short:** 28% are under 10 characters. Median English review ≈ 91 chars.
- **`thumbsUpCount` is almost always 0** — a few reviews go viral, most get no votes.
- **The `at` (date) column arrives as text** and must be converted to real dates before
  any time-based chart (`pd.to_datetime`).
- **Missing app-version info** is concentrated in certain periods/apps — explainable,
  harmless, kept.

---

## Step 4 — The language detour (an honest-measurement story)

We noticed non-English reviews and asked: *how many?* This turned into the best lesson of
the project so far — **don't trust a tool's output until you've checked it against the
actual data.**

1. A first quick check ("does the text contain non-Latin letters?") said ~17% — but that
   counted **emoji** as foreign text. Refined: only ~2% is genuinely non-Latin script
   (Arabic, Bengali, Chinese…).
2. We ran a proper language detector (`langdetect`) over every review. Raw result:
   only 61.7% English, with **Somali as the #2 language (8.1%)**. Suspicious!
3. We *read the actual "Somali" reviews*. They were `'good'`, `'so good'`, `'too slow'` —
   plain English, median length 4 characters. **The detector guesses wildly on short text.**
4. Recomputed using only reviews long enough to judge (≥40 chars): **~96.5% English.**
   True non-English ≈ 3–4% (Indonesian, Bengali, Arabic, Russian…).

**Decision:** don't filter any language out. The `language` label is saved per review
(in `master_clean_lang.parquet`) as a switch we can flip later. Instead, we chose an
embedding model that *understands* many languages (next step), so non-English reviews are
handled rather than discarded.

**Known honest gap:** Hindi/Tamil/etc. written in English letters ("bahut accha app hai")
gets mislabeled as English by every simple detector. We accept and document this for v1.

---

## Step 5 — Embeddings (notebook: 03.pipeline, Stage 1)

**The problem:** computers can't compare *meanings* of sentences directly.

**The solution:** an **embedding model** — a pre-trained neural network that converts any
sentence into a list of **384 numbers** (a *vector*). The magic property: sentences with
similar meaning get numerically *close* vectors, even if they share no words. "App keeps
crashing" and "it force-closes constantly" land near each other.

Think of it as giving every review a precise address in a 384-dimensional "meaning space,"
where the neighbourhood you live in is determined by what you're talking about.

- Model used: `paraphrase-multilingual-MiniLM-L12-v2` — pre-trained, multilingual (so a
  Spanish complaint lands near the same English complaint), free, runs on your laptop.
- **This is where the `len < 10` rule bites:** "good 😊" contains almost no meaning to
  locate in space, so reviews under 10 characters are excluded *from this stage on*.
- We prototyped on **10,000 random Netflix reviews** (not all 92k) — get the machinery
  working fast, scale up later by changing one variable (`SAMPLE_SIZE = None`).
- **Proof it works:** we picked a review and asked for its nearest neighbours in meaning
  space. Query: *"v6.16 crashes a lot…"* → neighbours: *"full of bugs"*, *"it crashed
  eventually"*, *"latest version is full of bugs"*. The model genuinely groups by meaning.

**Jargon:**
- **Vector / embedding** — the list of numbers representing one review's meaning.
- **Cosine similarity** — a 0-to-1 score of how close two vectors point; ~0.7+ means
  "clearly about the same thing" in our data.
- **Batch** — the model processes 64 reviews at a time for speed, not one by one.

---

## Step 6 — Clustering (notebook: 03.pipeline, Stage 2)

**Goal:** group the 10,000 vectors into **topics** — nobody tells the algorithm what the
topics are; it discovers groups of reviews that sit close together in meaning space.
That's **unsupervised learning** (no right answers given, structure is found, not taught).

It's a two-tool pipeline:

### Tool 1: UMAP — squash 384 dimensions down to 5
Clustering algorithms get confused in 384-dimensional space — with that many dimensions,
*everything* looks roughly equally far from everything else (the "curse of
dimensionality"). UMAP compresses the vectors from 384 numbers to 5, working hard to keep
neighbours as neighbours — like flattening a globe into a map while keeping nearby cities
nearby.

### Tool 2: HDBSCAN — find the dense groups
HDBSCAN looks for *dense* regions in the 5-number space: lots of reviews packed tightly
together = a topic. Its two superpowers over simpler methods:
- **We don't pick the number of topics** — it finds however many exist.
- **It's allowed to say "this review belongs to no topic"** — the label `-1`, called
  **noise**. Generic filler ("good app", "nice") lands there instead of polluting topics.

Settings we chose: a topic must have at least **50 reviews** (`min_cluster_size=50`) —
we don't care about 5-person quirks.

### The bug we hit (and the lesson)
First run crashed: *"Found array with 0 sample(s)"*. Diagnosis: UMAP's default starting
arrangement ("spectral init") failed silently and produced **NaN** (not-a-number) for
every row; HDBSCAN silently threw away all corrupted rows and had nothing left.
Fix: one argument, `init="random"`, in the UMAP call.

**Lesson worth keeping:** the error message pointed at HDBSCAN, but the *cause* was
upstream in UMAP. When code fails, check what each stage actually produced (we printed
`np.isnan(...).any()`) instead of trusting the last error line.

### First real result (Netflix, 10k sample)
```
7 clusters found, 4.6% noise
sizes: two huge blobs (6115 + 2930) + five small specific topics (58–167 each)
```
The blobs are probably broad "positive" vs "negative" masses. Useful topics need finer
splitting — that's **tuning**, the next thing we do. Knobs available: `min_cluster_size`
(smaller = more, finer topics), `min_samples` (higher = stricter, more noise), or
re-clustering *inside* a blob.

---

## Where we are / what's next

- [x] Sample & merge (500k)
- [x] Clean (495,967)
- [x] EDA
- [x] Language measured; decision: keep all, use multilingual models
- [x] Stage 1: embeddings work (verified by nearest-neighbour check)
- [~] Stage 2: clustering runs; needs tuning to split the two giant blobs
- [ ] Stage 3: label each cluster (top words + representative reviews)
- [ ] Stage 4: sentiment per review (off-the-shelf model)
- [ ] Stage 5: LLM summary (executive summary, pain points, feature requests)
- [ ] Scale from Netflix-10k to all five apps
- [ ] Wrap as the actual product flow (CSV in → report out)
