# ReviewIQ — Presentation Q&A (answers as of end of Day 2)

Prepared answers to the six check-in questions. Every number in here was actually
measured on this project — sources noted throughout.

---

## 1. Business problem — what does the product solve?

Product Managers of consumer apps receive **thousands of app-store reviews a month**.
The feedback they need is in there — bugs, billing complaints, feature requests — but
reading reviews manually doesn't scale, and skimming a few dozen gives a biased picture
(the loudest voices, not the biggest issues).

**ReviewIQ** turns a CSV of raw customer reviews into a decision-ready report in minutes:
an executive summary, ranked pain points, feature requests, pros/cons, sentiment split,
and topic clusters — each backed by counts, average ratings, and real example quotes.
The PM goes from "600 unread reviews" to "my top 3 problems this month are X, Y, Z,
affecting this share of reviewers, and here's what users literally say."

**We are not training a model.** The product composes pre-trained components
(embeddings, clustering, off-the-shelf sentiment) — which means it works on any app's
reviews out of the box, with no labeled training data required from the customer.

## 2. The dataset (origin, quality, EDA)

**Origin:** 5 public Kaggle datasets of Google-Play reviews: ChatGPT (1.05M), TikTok
(484k), Facebook (356k), Snapchat (283k), Netflix (154k). Used **only to develop and
validate** the pipeline — production input is whatever CSV a PM uploads.

**Sampling:** 100,000 random reviews per app (equal per app so no single app dominates;
`random_state=42` everywhere → fully reproducible), merged to 500,000 rows.

**Quality issues found and handled:**
- 4,018 duplicate `reviewId`s (same review scraped twice, sometimes with edited text) →
  deduplicated on `reviewId` only, never on text (thousands of people legitimately write
  just "good").
- 16 rows with no review text → dropped. Missing app-version fields → kept (only missing
  content or score justifies dropping). Final: **495,967 reviews**.

**Key EDA findings:**
- Ratings are **U-shaped** (mass at 1★ and 5★). That's real reviewer behaviour — people
  review when angry or delighted — so we deliberately did NOT rebalance it.
- Reviews are short: 28% under 10 characters; median English review ≈ 91 chars.
- **Language:** ~96% of substantive reviews are English. We measured this the hard way:
  the langdetect library initially claimed 8% of our corpus was Somali — reading those
  "Somali" reviews showed plain English like "so good" (the tool guesses wildly on short
  text). Lesson demonstrated: never trust tool output without checking it against raw data.
  Decision: keep all languages, use multilingual models instead of filtering.

## 3. Baseline models and improvements

The pipeline has two model choices; both were validated against simpler baselines on the
Netflix 10k prototype.

**Sentiment — baseline: VADER (rule-based, no ML):**
| Model | 1★ reviews called negative | 5★ reviews called positive |
|---|---|---|
| VADER baseline | 51% | 88% |
| **Multilingual transformer (ours)** | **73%** | 87% |

Same 300-review benchmark. VADER misses almost half the negative reviews (slang,
misspellings, non-English text defeat a fixed English lexicon); the distilled
multilingual transformer (`distilbert-...-sentiments-student`) finds +22 points more of
them at equal positive accuracy.

**Topic discovery — baseline: KMeans (k=31):**
| Approach | Silhouette score |
|---|---|
| KMeans directly on raw 384-dim embeddings | 0.035 |
| KMeans on UMAP-reduced 5-dim | 0.341 |
| **UMAP + HDBSCAN (ours)**, non-noise points | **0.342** |

Two findings: (a) dimensionality reduction is what creates clusterable structure
(0.035 → 0.341 — the "curse of dimensionality" made visible); (b) at equal silhouette,
HDBSCAN wins on *usefulness*: it discovers the number of topics itself (no guessing k),
and it isolates 26% generic filler ("good app") as noise instead of smearing it across
every topic — KMeans' largest clusters are unreadable mixed bags, HDBSCAN's are
single-issue (login failures, billing, subtitle bugs).

**Embedding quality check:** nearest-neighbour inspection — "v6.16 crashes a lot" retrieves
"full of bugs", "it crashed eventually", "latest version is full of bugs". Meaning-based
grouping works, including across languages (multilingual model).

## 4. Which metric, and why?

There is no single metric for an unsupervised pipeline — we chose one per stage, plus a
rule about how to use them:

- **Sentiment: agreement with star ratings as proxy ground truth.** Reviews arrive
  pre-labeled by their authors (1★ vs 5★). A good text-sentiment model must broadly agree
  (ours: 73% / 88%, with a clean monotonic staircase across 2–4★). Disagreements are
  often *signal*, not error — "5 stars but please fix subtitles" is exactly what the
  product exists to surface.
- **Clustering: silhouette score + noise share + human coherence.** Silhouette (how
  tight/separated clusters are) for comparability; noise % because a review corpus is
  full of genuine filler; and reading the clusters, because a "good" silhouette with
  unreadable topics is worthless to a PM. We tuned by reading, and report all three.
- **Why not accuracy/F1 everywhere:** no labeled topics exist — this is unsupervised
  discovery. Inventing labels to score against would just launder our own assumptions.

## 5. Timeline and where we are

One-week sprint to demo (updated Day 5):

| Day | Plan | Status |
|---|---|---|
| 1 | Tune clustering | ✅ 31 topics, 26% noise |
| 2 | Label clusters + sentiment | ✅ validated (73%/88% staircase) |
| 3 | Report generator (template-based) | ✅ reports/netflix_report.md |
| 4 | Pipeline → reusable module (src/reviewiq.py) | ✅ smoke test caught 2 bugs pre-launch |
| 5 | Full-scale run: all 5 apps, ~350k reviews | ✅ 4/5 apps complete, TikTok in flight; 100-review product scenario proven (~1 min CSV→report) |
| 6 | Read all 5 reports, end-to-end rehearsal | ← next |
| 7 | Buffer + presentation prep | buffer intact |

Everything upstream (sampling, cleaning, EDA, embeddings) was complete before Day 1.
Currently **~1 day ahead of plan** — Day 4's LLM stage was replaced by the
template-based decision (#10), which bought the schedule margin.

## 6. Future work and expected difficulties

**Future work:**
- **LLM polish layer** (optional plug-in): rewrite the template report in fluent prose;
  design keeps it non-blocking — no API key, still a full report.
- **Romanized-language detection:** "Hinglish" (Hindi in Latin letters) evades every
  simple language detector. Our clustering already quarantines it (a 145-review cluster
  of pure Hinglish emerged by itself); proper handling needs e.g. fastText language ID.
- **Full multilingual reports:** per-language topic summaries for global apps.
- **Product hardening:** CSV-upload wrapper → simple UI (Streamlit) → per-upload compute
  budget (100 reviews ≈ seconds, so real-time is feasible).
- **Weighting by `thumbsUpCount`** so a complaint upvoted 3,000 times outranks a one-off.

**Known difficulties:**
- **CPU-only compute:** embedding 345k reviews takes hours on a laptop → full-scale runs
  scheduled overnight; production would use a small GPU or batch API.
- **Short reviews** (28% under 10 chars) carry no clusterable meaning — excluded from
  topics by rule, but still counted in sentiment/statistics.
- **Very unequal cluster granularity per app** — each app needs its own noise/size
  trade-off; settings that give Netflix 31 clean topics may need re-tuning elsewhere.
- **Honest limits:** sentiment ~73% on angry text, Hinglish passes as English — both
  documented rather than hidden; the report shows counts and real quotes so a PM can
  always verify claims against the underlying reviews.
