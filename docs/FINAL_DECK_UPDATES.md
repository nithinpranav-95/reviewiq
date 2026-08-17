# Final-presentation deck updates (from the midterm REVISED deck)

Every number below is measured and reproducible (notebooks 03–05, app.py).

---

## Slide 1 — Title
- Change "MIDTERM PRESENTATION" → **"FINAL PRESENTATION"**.

## NEW SLIDE (after slide 6) — "You asked us to train a model. We did."

> **Transfer learning:** a logistic-regression head (385 trained weights) on top of the
> frozen pretrained embeddings (118M weights). Trains in ~2 minutes on CPU.
>
> - Netflix hold-out test: **0.897 accuracy** — train = test, so no overfitting
> - Gradient boosting (XGBoost family) tied it (0.890) → kept the simpler model
> - **Leave-one-app-out** (train on 4 apps, test on the unseen 5th):
>
> | Held-out app | Trained | Pretrained |
> |---|---|---|
> | ChatGPT | **0.932** | 0.899 |
> | Facebook | **0.859** | 0.825 |
> | Netflix | **0.875** | 0.827 |
> | Snapchat | **0.867** | 0.827 |
> | TikTok | **0.832** | 0.796 |
>
> Trained wins on every unseen app (+3–5 pts) — **but** the yardstick is star-agreement,
> which the trained model directly optimizes. The decisive test (hand-labeled
> text-vs-star disagreements) is future work.
>
> **Product conclusion:** ship pretrained for the works-on-any-app promise;
> fine-tune per customer as the upgrade path.

*(Spoken line: "the metrics favor our trained model — and we can explain exactly why
that scoreboard is partially rigged in its favor. That's the difference between
measuring and assuming.")*

## Slide 7 — "Current Progress" → retitle "Since the midterm"

> ✅ **Trained our own models** — transfer learning, validated vs the pretrained (see new slide)
> ✅ **Built the dashboard** — explore all 5 apps, upload a CSV, or **paste any
>    Google-Play link → live report in ~1 minute** (the midterm roadmap's
>    "App Store Integration", delivered)
> ✅ Self-healing clustering after a silent failure at scale (TikTok: 2 blobs → 94 topics)
> 🔜 Deploy to Hugging Face Spaces — public URL in the repo README

Progress bars: Pipeline 100% · Dashboard 100% · Deployment 0% (next step)

## Slide 8 — Future Roadmap (update: one item is DONE)
- **Remove** "App Store Integration" (delivered — it's in the demo).
- Keep: Interactive dashboard → change to "Deploy dashboard publicly (Hugging Face Spaces)".
- Keep: Automated monitoring → upgrade wording: "**Monitoring agent** — watches new
  reviews weekly, flags emerging complaint clusters, drafts the ticket."
- Add: "Aspect-based sentiment (per-theme, not per-review) — how Chattermill-class
  tools handle '5★ but fix subtitles'."
- Add: "Hand-labeled evaluation set — the honest tie-breaker for trained vs pretrained."

## Slide 10 — Key takeaways (adjust two bullets)
- Add to pipeline bullet: "…and a live product: **any Play-Store link → report in ~1 min**."
- Replace the sentiment bullet with: "**Measured, not assumed:** baselines for every
  choice; trained our own model and benchmarked it honestly against the pretrained."

## Demo plan (not a slide — the run of show)
1. Explore tab: Netflix report (safe, instant, impressive numbers).
2. The stunt: ask the audience for an app → paste its Play-Store link → talk through
   the pipeline stages while the spinner runs (~1–2 min) → read its pain points aloud.
3. Backups, in order: `demo_upload.csv` in the upload tab (no scraping needed) →
   Explore tab (no compute at all) → screenshots (no app at all).
4. Tech check beforehand: dashboard starts (`cd` to project first!), WiFi works,
   screenshots on the Desktop.
