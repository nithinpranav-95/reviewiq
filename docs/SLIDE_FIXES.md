# Slide-by-slide fixes for the midterm deck

Paste-ready replacement text. Only changed slides are listed — slides 1, 2, 5, 8, 10
keep their structure (slides 2/5/10 just gain a real number, marked ➕).
Every number below is measured and reproducible in the repo (notebooks 01–04,
docs/PRESENTATION_QA.md).

---

## Slide 2 — The Problem (keep, add one number ➕)

At the end of "The Challenge", add:

> Our development corpus alone: **495,967 reviews across five apps** — no human
> could read them; ReviewIQ digests an app's worth in minutes.

---

## Slide 3 — Dataset & Preprocessing (REPLACE the "Preprocessing Pipeline" list)

**Delete:** "Tokenization & stop-word removal / Lemmatization for normalization"
*(we never did these — modern sentence embeddings take raw text, and that's a
feature: no information is destroyed before the model sees it)*

**Replace with:**

> **Preprocessing Pipeline**
> - Deduplication on review ID only — 4,018 re-scraped duplicates removed
>   (never dedupe on text: thousands of people legitimately write just "good")
> - Drop only rows missing review text or rating (16 rows) → **495,967 clean reviews**
> - Very short reviews (<10 characters — 28% of the corpus!) kept in the data,
>   excluded only at the embedding stage where they carry no meaning
> - Language measured per review: ~96% English among substantive reviews —
>   we keep all languages and use multilingual models instead of filtering

---

## Slide 4 — EDA (REPLACE two bullets, fix one phrase)

**Delete:** "Feature Correlation — Heatmap identifies relationships…" *(never made)*
**Delete the phrase:** "before training began" *(we train nothing — pretrained
components only; this is a core design decision, don't contradict it)*

**Replace the four EDA cards with:**

> **Rating distribution is U-shaped** — mass at 1★ and 5★. People review when
> angry or delighted. We kept this skew deliberately: it's real signal.
>
> **Reviews are short** — 28% under 10 characters; median English review ≈ 91 chars.
> Length rules decide what can be clustered.
>
> **The language trap** — our detector initially claimed 8% of reviews were *Somali*.
> Reading them showed plain English ("so good") — the tool guesses wildly on short
> text. Measured properly: ~96% English. Lesson: never trust tool output unchecked.
>
> **Engagement is extreme** — most reviews get 0 thumbs-up; a handful go viral.

---

## Slide 6 — Evaluation Metrics (REPLACE the verdict — current claim is not true)

**Delete:** "UMAP + HDBSCAN consistently outperformed the K-Means baseline across
all three metrics" *(we measured it: one win, one tie, one loss)*

**Replace with (keep the three metric cards, then):**

> **Measured on Netflix 10k (all reproducible in notebook 03):**
> | | K-Means | UMAP + HDBSCAN |
> |---|---|---|
> | Silhouette (higher better) | 0.341 | 0.342 — tie |
> | Davies–Bouldin (lower better) | 1.016 | **0.688** ✅ |
> | Calinski–Harabasz (higher better) | **10,136** | 2,428 |
>
> **The metrics disagree — which is exactly why manual inspection matters.**
> Calinski–Harabasz rewards the big, equal-sized clusters K-Means produces *by
> force*. HDBSCAN's decisive advantages don't show in these scores: it discovers
> the number of topics itself, and it isolates ~26% generic filler ("good app")
> as noise instead of smearing it into every topic.
>
> Bonus finding: K-Means directly on raw 384-dim embeddings scores **0.035** —
> dimensionality reduction is what creates clusterable structure at all
> (the curse of dimensionality, demonstrated on our own data).

---

## Slide 7 — Current Progress (UPDATE — you are further along than the slide says)

**Replace the status blocks with:**

> ✅ **Completed** — Data pipeline (495,967 reviews), EDA, embeddings, clustering
> tuned & validated against baselines, sentiment validated (73% / 88% agreement
> with 1★/5★ ratings vs 51% for rule-based baseline), template report generator,
> **full-scale reports for all five apps**, and a working product flow:
> a raw 100-review CSV → finished report in ~1 minute.
>
> 🔄 **In Progress** — Final polish: per-app tuning hardening, documentation,
> end-to-end rehearsal.
>
> 🔲 **Remaining** — Interactive dashboard (Streamlit), deployment, presentation.

**Progress bars:** Overall ~75–80% · Data & NLP pipeline 100% · Dashboard & UI 0–20%

---

## Slide 9 — Challenges (REPLACE the generic list with what actually happened)

**Delete:** "Obtaining large, high-quality labeled datasets" *(we need no labels —
nothing is trained; this bullet contradicts the architecture)*

**Replace with challenges we actually hit and solved:**

> - **Silent model failure at scale** — on TikTok (74k reviews) the clustering
>   collapsed into 2 giant blobs and produced an empty report. Fixed with a
>   self-healing check: detect degenerate clustering, auto-retry with finer settings.
> - **One-size-fits-all tuning doesn't exist** — each app's review style needs its
>   own clustering granularity; we scale settings by corpus size and verify.
> - **Language detection lies on short text** — and Romanized Hindi ("Hinglish")
>   passes as English; our clustering visibly quarantines it (a 145-review
>   Hinglish cluster emerged on its own).
> - **Noisy, informal text** — emoji, slang, misspellings; rule-based tools break
>   (VADER missed half the anger), multilingual transformers cope.
> - **CPU-only compute** — 350k reviews ≈ overnight; budgeted via caching every
>   stage to disk and resumable batch runs.

---

## Slide 10 — Key Takeaways (keep, add numbers ➕)

> **End-to-End Pipeline** — 495,967 raw reviews → five decision-ready reports.
> **Sentiment & Clustering** — beats rule-based baseline by +22 points on
> negative-review detection; clustering finds 31–94 coherent topics per app.
> **Business Value** — a PM's 100-review CSV becomes an executive report in
> ~1 minute, at zero API cost.

---

## Two Q&A grenades to be ready for (because the deck invites them)

1. *"Show me your lemmatization / tokenization."* → After the Slide 3 fix: "we
   deliberately don't — sentence embeddings consume raw text; destroying structure
   before the model sees it would only lose information."
2. *"What was your Davies–Bouldin?"* → After the Slide 6 fix: 0.688 vs 1.016,
   HDBSCAN better — and you can explain *why* the third metric disagrees. That
   answer is a flex, not a dodge.
