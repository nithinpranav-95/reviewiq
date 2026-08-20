"""
ReviewIQ pipeline engine.

Reusable functions for the full flow:
    gate -> embed -> reduce+cluster -> label topics -> sentiment -> report

Used two ways:
  1. Batch validation over the 5 Kaggle apps (run as a script, resumable).
  2. The product wrapper: run_pipeline(any_reviews_dataframe) -> report text.

All settings follow docs/HANDOFF.md decisions #4, #6, #7, #8, #9, #10.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Lazy model holders so importing this file stays instant.
_EMBEDDER = None
_SENTIMENT = None

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
SCALE_DIR = DATA / "full_run"


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def get_sentiment():
    global _SENTIMENT
    if _SENTIMENT is None:
        from transformers import pipeline
        _SENTIMENT = pipeline("sentiment-analysis", model=SENTIMENT_MODEL,
                              truncation=True)
    return _SENTIMENT


def gate(df):
    """Embedding gate: content length >= 10 (decision #4). No language filter (#7)."""
    out = df[df["content"].str.len() >= 10].reset_index(drop=True).copy()
    return out


def embed(texts, batch_size=64):
    model = get_embedder()
    return model.encode(list(texts), batch_size=batch_size,
                        show_progress_bar=False, convert_to_numpy=True)


def cluster(embeddings):
    """UMAP -> HDBSCAN (decision #8). min_cluster_size scales with corpus size
    so a 90k-review app doesn't shatter into hundreds of micro-topics
    (30 was tuned at 10k = 0.3% of the corpus; we keep that ratio, floor 30)."""
    import umap
    import hdbscan

    n = len(embeddings)
    # 0.3% of the corpus (ratio tuned at 10k), but never below 5:
    # small corpora (like a PM's 100-review upload) need small clusters
    # to find any structure at all.
    mcs = max(5, round(n * 0.003))
    ms = max(3, mcs // 6)

    reduced = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine",
                        random_state=42, init="random").fit_transform(embeddings)

    # The scaled settings can still be too coarse for a homogeneous corpus
    # (TikTok at 74k collapsed into 2 blobs). If the result is degenerate —
    # almost no clusters, or one cluster swallowing most reviews — retry
    # with finer settings before accepting it.
    for attempt in range(3):
        labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                 metric="euclidean").fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        biggest = max(np.bincount(labels[labels != -1]), default=0) if n_clusters else 0
        degenerate = n_clusters < 5 or (n > 1000 and biggest > 0.5 * n)
        if not degenerate or mcs <= 5:
            break
        mcs = max(5, mcs // 2)
        ms = max(3, mcs // 6)
        print(f"    degenerate clustering ({n_clusters} clusters, biggest {biggest}) "
              f"-> retrying with mcs={mcs}, ms={ms}", flush=True)

    return labels, {"min_cluster_size": mcs, "min_samples": ms}


def top_words(texts, n=6):
    """Raw-count fallback labeling (used only when there are too few clusters
    for distinctiveness scoring to work)."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words="english", min_df=2)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    totals = np.asarray(X.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    return [vocab[i] for i in totals.argsort()[::-1][:n]]


def distinctive_words(cluster_texts, n=6):
    """Label each cluster by its most DISTINCTIVE words, not its most frequent.

    Treat each cluster's combined text as one document and TF-IDF across those
    documents (the c-TF-IDF idea from BERTopic). Corpus-generic words that show
    up in every cluster ('app', the app's own name, 'playstore') get weighted
    down to ~0; words concentrated in one cluster ('password', 'subtitles')
    rise to the top. Input: {cluster_id: [texts]}. Returns {cluster_id: [words]}.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    ids = list(cluster_texts.keys())
    docs = [" ".join(cluster_texts[c]) for c in ids]

    # Only words appearing in >= 5 individual reviews may become labels —
    # otherwise one-off typos ("nextflix") look maximally "distinctive".
    all_reviews = [t for c in ids for t in cluster_texts[c]]
    try:
        counter = CountVectorizer(stop_words="english", min_df=5)
        counter.fit(all_reviews)
        allowed = counter.get_feature_names_out().tolist()
    except ValueError:          # tiny corpus: no word clears min_df
        allowed = None

    vec = TfidfVectorizer(stop_words="english", vocabulary=allowed,
                          sublinear_tf=True)
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    out = {}
    for row, c in enumerate(ids):
        weights = X[row].toarray().ravel()
        out[c] = [vocab[i] for i in weights.argsort()[::-1][:n] if weights[i] > 0]
    return out


def topic_table(df):
    """One row per cluster: size, avg score, top words, example."""
    clusters = [c for c in sorted(df["cluster"].unique()) if c != -1]

    # Distinctiveness scoring needs several clusters to compare against each
    # other; with very few, fall back to plain counts.
    if len(clusters) >= 3:
        texts_by_cluster = {c: df.loc[df["cluster"] == c, "content"].tolist()
                            for c in clusters}
        labels = distinctive_words(texts_by_cluster)
    else:
        labels = {c: top_words(df.loc[df["cluster"] == c, "content"].tolist())
                  for c in clusters}

    n_clustered = int((df["cluster"] != -1).sum())
    rows = []
    for c in clusters:
        sub = df[df["cluster"] == c]
        label = ", ".join(labels.get(c, []))
        # A cluster swallowing >20% of everything is generic chatter, not a
        # theme — say so instead of pretending its oddball words are a topic.
        if len(sub) > 0.2 * n_clustered and len(clusters) >= 5:
            label = f"(general / mixed feedback — {label})"
        rows.append({
            "cluster": c,
            "n_reviews": len(sub),
            "avg_score": round(sub["score"].mean(), 1),
            "top_words": label,
            "example": sub["content"].iloc[0][:80],
        })
    return pd.DataFrame(rows).sort_values("n_reviews", ascending=False)


def add_sentiment(df, batch_size=64, log_every=4000):
    p = get_sentiment()
    texts = df["content"].tolist()
    results = []
    for i in range(0, len(texts), batch_size):
        results.extend(p(texts[i:i + batch_size], batch_size=batch_size))
        if i % log_every < batch_size:
            print(f"    sentiment {i}/{len(texts)}", flush=True)
    df = df.copy()
    df["sentiment"] = [r["label"] for r in results]
    df["sentiment_conf"] = [round(r["score"], 3) for r in results]
    return df


REQUEST_WORDS = ["add", "please", "wish", "want", "should", "would be", "need"]


def build_report(df, topics, app_name):
    """Template report (decision #10): real numbers slotted into fixed sentences."""
    n = len(df)
    sent_pct = (df["sentiment"].value_counts(normalize=True) * 100).round(0)

    def request_share(cluster_id):
        texts = df.loc[df["cluster"] == cluster_id, "content"].str.lower()
        return texts.apply(lambda t: any(w in t for w in REQUEST_WORDS)).mean()

    topics = topics.copy()
    topics["request_share"] = topics["cluster"].apply(request_share)

    pain = topics[topics["avg_score"] <= 2.5].sort_values("n_reviews", ascending=False)
    pros = topics[topics["avg_score"] >= 4.0].sort_values("n_reviews", ascending=False)
    requests = topics[topics["request_share"] >= 0.5].sort_values(
        "request_share", ascending=False)

    def name(r):
        return " / ".join(r.top_words.split(", ")[:2])

    def best_quote(cluster_id):
        cands = df.loc[(df["cluster"] == cluster_id) &
                       (df["sentiment"] == "negative"), "content"].head(20)
        if len(cands) == 0:
            cands = df.loc[df["cluster"] == cluster_id, "content"].head(20)
        return max(cands, key=len)[:140]

    summary = (f"Of {n:,} reviews analyzed, {sent_pct.get('negative', 0):.0f}% are "
               f"negative, {sent_pct.get('positive', 0):.0f}% positive.")
    if len(pain):
        summary += (" The biggest pain points are: "
                    + "; ".join(f"**{name(r)}** ({r.n_reviews} reviews, avg {r.avg_score}★)"
                                for r in pain.head(3).itertuples()) + ".")
    else:
        summary += (" No single dominant pain-point theme emerged; "
                    "negative feedback is spread across topics.")

    lines = [f"# ReviewIQ Report — {app_name.title()}\n",
             f"*Based on {n:,} reviews analyzed*\n",
             "## Executive summary\n",
             summary]

    lines.append("\n## Top pain points\n")
    if len(pain) == 0:
        lines.append("*(no cluster averaged 2.5★ or below)*")
    for r in pain.head(5).itertuples():
        lines.append(f"- **{r.top_words}** — {r.n_reviews} reviews, avg {r.avg_score}★")
        lines.append(f"  > \"{best_quote(r.cluster)}\"")

    lines.append("\n## What users love\n")
    if len(pros) == 0:
        lines.append("*(no cluster averaged 4.0★ or above)*")
    for r in pros.head(3).itertuples():
        lines.append(f"- **{r.top_words}** — {r.n_reviews} reviews, avg {r.avg_score}★")

    lines.append("\n## Feature requests\n")
    if len(requests) == 0:
        lines.append("*(no topic where most reviews ask for something)*")
    for r in requests.head(3).itertuples():
        lines.append(f"- **{r.top_words}** — {r.request_share:.0%} of its "
                     f"{r.n_reviews} reviews are asking for something")

    return "\n".join(lines)


def run_pipeline(df, app_name, out_dir=None, embeddings=None):
    """Full flow for one set of reviews. Returns (clustered_df, topics, report_text).
    If out_dir is given, saves all artifacts there (resumable batch use)."""
    df = gate(df)
    print(f"  {app_name}: {len(df)} reviews after gate", flush=True)

    if embeddings is None:
        embeddings = embed(df["content"])
        print(f"  {app_name}: embedded {embeddings.shape}", flush=True)

    labels, params = cluster(embeddings)
    df = df.copy()
    df["cluster"] = labels
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).mean() * 100
    print(f"  {app_name}: {n_topics} topics, {noise:.0f}% noise (params {params})", flush=True)

    df = add_sentiment(df)
    topics = topic_table(df)
    report = build_report(df, topics, app_name)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{app_name}_embeddings.npy", embeddings)
        df.to_parquet(out_dir / f"{app_name}_clustered.parquet", index=False)
        topics.to_parquet(out_dir / f"{app_name}_topics.parquet", index=False)
        topics.to_csv(out_dir / f"{app_name}_topics.csv", index=False)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"{app_name}_report.md").write_text(report, encoding="utf-8")
        print(f"  {app_name}: saved artifacts + report", flush=True)

    return df, topics, report


def main():
    """Batch run over all 5 apps. Resumable: skips an app whose report exists."""
    master = pd.read_parquet(DATA / "master_clean_lang.parquet")
    for app in sorted(master["app_name"].unique()):
        # Resume marker: the batch's OWN artifact, not the report file.
        # (A report generated elsewhere — e.g. the notebook's 10k prototype —
        # once made the batch silently skip a full-scale run.)
        done_marker = SCALE_DIR / f"{app}_topics.parquet"
        if done_marker.exists():
            print(f"SKIP {app} (full-run artifacts exist)", flush=True)
            continue
        print(f"=== {app} ===", flush=True)
        run_pipeline(master[master["app_name"] == app], app, out_dir=SCALE_DIR)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
