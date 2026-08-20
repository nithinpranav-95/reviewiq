# ReviewIQ — complete final-presentation content
# One slide per section. Bullets = on-slide text (max 4). Italic = speaker notes (say, don't show).
# All numbers are measured and reproducible in the repo.

---

## 1 · Title

**ReviewIQ — AI-Powered Review Intelligence**
Sentiment Analysis & Review Summarization using NLP
Data Science & AI Bootcamp — FINAL PRESENTATION
**Nithin & Neel**

*"We're Nithin and Neel. ReviewIQ reads thousands of app-store reviews and turns them into a report a product manager can act on. Today: what we built, how we proved it works, and what it taught us."*

---

## 2 · The problem

**Thousands of reviews. Nobody reads them.**

- Popular apps receive thousands of reviews weekly — the answers to "what should we fix?" are buried inside
- Manual reading doesn't scale; skimming hears the loudest voices, not the biggest issues
- Our stakeholder: the product manager deciding what to build next
- ReviewIQ: reviews in → executive summary, ranked pain points with real quotes, feature requests, sentiment — in about a minute

*"Our development corpus alone is 495,967 reviews. No human reads that. Our pipeline does."*

---

## 3 · The data

**495,967 real reviews, five real apps**

- Netflix, ChatGPT, TikTok, Snapchat, Facebook — public Kaggle datasets (Google Play)
- 100,000 random reviews per app, fixed seed → fully reproducible
- Cleaning: dedupe on review-ID only (4,018 re-scraped duplicates), drop only rows missing text or rating
- Kept deliberately: very short reviews (28% under 10 characters!) and all languages (~96% English, measured)

*Story to tell: "Our language detector claimed 8% of reviews were Somali. We read them — they said 'so good'. The tool guesses wildly on short text. Lesson: never trust a tool's output without checking the raw data. That lesson shaped the whole project."*

---

## 4 · What the data told us (EDA)

**Three patterns that drove the design**

- Ratings are U-shaped: people review when angry or delighted — we kept the skew (it's real behaviour, not noise)
- Reviews are short: median ~91 characters; 28% under 10 — too short to carry meaning
- Text disagrees with stars: "5★ but please fix subtitles" — the text holds what a PM can act on
- No model training anywhere — pretrained components + unsupervised discovery, so it works on any app out of the box

---

## 5 · How it works (the pipeline)

**Review text → meaning coordinates → topics → sentiment → report**

- Embeddings: a pretrained multilingual model (MiniLM) turns each review into 384 numbers — similar meaning = nearby points, even across languages
- UMAP squashes 384 → 5 dimensions; HDBSCAN finds dense groups = topics, and quarantines generic filler ("good app 👍") as noise
- A second pretrained model (DistilBERT) judges each review's sentiment from the text — independent of its stars
- The report is templates: real counts + real quotes slotted in — zero API cost, zero hallucination risk

*Tech stack (say it): "All Python — pandas, sentence-transformers, Hugging Face, UMAP, HDBSCAN, scikit-learn. Both neural models pretrained, running locally on a laptop."*

---

## 6 · Proving it: baselines and metrics

**Every choice measured against a simpler alternative**

- Sentiment vs rule-based VADER: it catches 51% of 1-star anger; our transformer catches **73%** at equal positive accuracy (88%)
- Clustering vs K-Means: silhouette **tie** (0.342 vs 0.341), Davies–Bouldin **win** (0.688 vs 1.016), Calinski–Harabasz **loss** — a metric that rewards K-Means' forced equal clusters
- The knockout number: K-Means on raw 384-dim scores **0.035** — dimensionality reduction is what creates structure at all
- Metrics disagree → inspection decides: our clusters read as single issues — login, billing, subtitle bugs

*"There's no accuracy score for unsupervised discovery — no labels exist. Honest validation is: comparable metrics, noise share, and reading your own clusters."*

---

## 7 · You asked us to train a model. We did.

**Transfer learning: a 385-weight head on 118M frozen pretrained weights**

- Hold-out test (Netflix, 80/20): **accuracy 0.897 — train = test, no overfitting**; ~0.90 precision & recall
- Gradient boosting (XGBoost family) tied it (0.890) → kept the simpler model
- Leave-one-app-out (train on 4, test on unseen 5th): **trained wins all five apps (+3–5 pts)** — ChatGPT 0.932 vs 0.899, Netflix 0.875 vs 0.827…
- The honest caveat: the yardstick is star-agreement, which the trained model directly optimizes; decisive test = hand-labeled disagreements (future work)

*"Conclusion: ship pretrained for the any-app promise; fine-tune per customer as the upgrade path. We measured the trade instead of assuming it."*

---

## 8 · When models fail silently (the TikTok story)

**The bug that taught us the most**

- At 74,000 reviews, TikTok's clustering silently collapsed into 2 giant blobs → the report came out empty
- No crash, no error — the run even looked "clean" (1% noise)
- Fix: a self-healing check — detect degenerate clustering, automatically retry with finer settings
- Same data after the fix: **94 coherent topics** — account bans and broken updates on top, exactly where TikTok's real pain is

*"Code review can't catch this. Validation — reading your own output — can."*

---

## 9 · Since the midterm

**Everything on the midterm roadmap that said "next" — done**

- Trained and benchmarked our own sentiment models (previous slide)
- Built the dashboard: explore all 5 apps, upload a CSV, or **paste any Play-Store link → live report in ~1 minute**
- That last one was the midterm's "App Store Integration" future-work item — delivered, you'll see it live
- Self-healing clustering, launcher, docs — pipeline and dashboard are feature-complete

---

## 10 · LIVE DEMO (not a slide — the run of show)

1. Explore tab → Netflix: 93k reviews, 30 topics, pain points with quotes (instant, no compute)
2. The stunt: "name any app" → paste its Play-Store link → talk the pipeline while the spinner runs (~1–2 min) → read its pain points aloud
3. Backups, in order: demo_upload.csv (no scraping) → Explore tab (no compute) → screenshots (no app)

*While the spinner runs: "Right now it's embedding 200 reviews into meaning-space, discovering the topics, scoring sentiment, and writing the report — on this laptop, no cloud, no API."*

---

## 11 · Future roadmap

**What's genuinely next (not what we ran out of time for)**

- Deploy publicly: Hugging Face Spaces — a URL anyone can open (dashboard is deployment-ready)
- Monitoring agent: watches new reviews weekly, flags emerging complaint clusters, drafts the ticket
- Aspect-based sentiment: per-theme verdicts ("delivery fast ➕, support ignored me ➖") — the mature answer to text-vs-stars
- Apple App Store ingestion + hand-labeled evaluation set

---

## 12 · Key takeaways

**ReviewIQ turns review noise into a to-do list**

- 495,967 reviews → five decision-ready reports + a live product: any Play-Store link → report in ~1 minute, $0 per report
- Measured, not assumed: baselines for every choice; trained our own model and benchmarked it honestly against the pretrained
- Biggest lesson: models fail silently — validation and reading your data beat fancier algorithms
- **Questions?**

*(Appendix slides for Q&A: docs/TECH_SLIDES_CONTENT.md — metrics deep-dives, leave-one-app-out table, the "what we deliberately did NOT do" slide.)*

---

# Presentation rules compliance check
- Names + project + plain-English intro on slide 1 ✓
- Start with why ✓ · stakeholder named ✓ · before/after comparisons (VADER vs ours, K-Means vs ours, TikTok before/after) ✓
- Tech stack spoken at slide 5 ✓ · raw numbers throughout ✓ · stories (Somali, TikTok) ✓
- 11 content slides + demo ≈ 8–10 min at ~40s/slide + 2-min demo — tight but legal; compress slides 3–4 if over
- Live demo of working software with layered backups ✓
