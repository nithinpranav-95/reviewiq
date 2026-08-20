# Technical slide content — paste into any slide tool
# (designed as Q&A / appendix slides: one idea per slide, max 4 points, numbers exact)

---

## SLIDE: How we measure topic quality

**Three metrics + human inspection (there is no "accuracy" without labels)**

- Silhouette — is each review closer to its own cluster than the next one? **0.342 vs 0.341 (K-Means): tie**
- Davies–Bouldin — cluster blur vs separation (lower better): **0.688 vs 1.016 — we win**
- Calinski–Harabasz — favors K-Means' forced equal clusters by design: **K-Means wins**
- Metrics disagree → inspection decides: our clusters read as single issues (login, billing, subtitles)

*Speaker note: "The metric K-Means wins is the one structurally biased toward K-Means. That's why no single number decides unsupervised quality."*

---

## SLIDE: The curse of dimensionality — measured on our own data

- K-Means directly on raw 384-dim embeddings: **silhouette 0.035** — no structure
- Same K-Means after UMAP reduction to 5-dim: **0.341** — 10× jump
- Reduction doesn't lose the signal — it *creates* clusterable structure
- HDBSCAN adds what K-Means can't: finds the topic count itself + isolates ~26% generic filler ("good app 👍") as noise

*Speaker note: "The failed run that reported 1% noise produced garbage; 26% noise is what honesty looks like on app reviews."*

---

## SLIDE: Sentiment — validated against a baseline, not assumed

**Yardstick: agreement with the author's own star rating (300-review benchmark + full 10k)**

- Rule-based baseline (VADER, word list): catches only **51%** of 1-star anger
- Pretrained multilingual transformer: **73%** on 1-star, **88%** on 5-star — **+22 points**
- Clean staircase across 2–4 stars → holds at scale (10,000 reviews)
- Where text disagrees with stars ("5★ but fix subtitles") — that's signal, not error: it's what the product surfaces

---

## SLIDE: You asked us to train a model. We did. (transfer learning)

- Logistic-regression head: **385 trained weights** on top of **118M frozen pretrained weights** — trains in ~2 min on CPU
- Hold-out test (Netflix, 80/20 stratified): **accuracy 0.897 — train = test, no overfitting**
- Precision & recall ≈ **0.90** per class
- Gradient boosting (XGBoost family) tied it (0.890) → kept the simpler model

*Speaker note: "The intelligence is in the embeddings; the classifier just draws a boundary through meaning-space."*

---

## SLIDE: Leave-one-app-out — does training generalize?

**Train on 4 apps, test on the 5th it never saw:**

| Held-out | Trained | Pretrained |
|---|---|---|
| ChatGPT | **0.932** | 0.899 |
| Facebook | **0.859** | 0.825 |
| Netflix | **0.875** | 0.827 |
| Snapchat | **0.867** | 0.827 |
| TikTok | **0.832** | 0.796 |

- Trained wins all 5 (+3–5 pts) — **but** the yardstick is star-agreement, which the trained model directly optimizes
- Decisive test: hand-labeled disagreement cases — future work
- Product conclusion: ship pretrained (works on any app, zero setup); fine-tune per customer as upgrade path

---

## SLIDE: When models fail silently (the TikTok story)

- At 74,000 reviews, clustering collapsed into **2 giant blobs** → the report came out empty
- No error, no crash — the metrics even looked "clean" (1% noise)
- Fix: a **self-healing check** — detect degenerate clustering, auto-retry with finer settings
- Same data after the fix: **94 coherent topics**

*Speaker note: "Code review can't catch this. Validation — reading your own output — can. Biggest lesson of the project."*

---

## SLIDE: What we deliberately did NOT do (and why)

- **No manual text preprocessing** (no stop-word removal / lemmatization before models) — transformers tokenize internally; stripping "not" from "not good" destroys meaning
- **No language filtering** — measured ~96% English; kept all languages, used multilingual models (after catching our detector calling short English reviews "Somali")
- **No training in the product** — pretrained + unsupervised = works on any app, in ~50 languages, the moment you paste a link
- **No LLM in the report** — templates with real counts and quotes: zero cost, zero hallucination risk

---

## SLIDE: The numbers that matter

- **495,967** cleaned reviews (5 apps, deduped, seeded → fully reproducible)
- **384** dimensions per review ("coordinates of meaning") → reduced to 5 for clustering
- **30–94 topics** per app · ~26–35% honest noise
- **~1 minute**: any Play-Store link → finished report · **$0** per report (all local, no APIs)
