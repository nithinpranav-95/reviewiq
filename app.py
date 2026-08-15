"""
ReviewIQ dashboard — Streamlit UI over the pipeline in src/reviewiq.py.

Run from the project root:
    streamlit run app.py

Two tabs:
  1. Explore apps  — precomputed full-scale results for the 5 Kaggle apps (instant).
  2. Upload CSV    — run the live pipeline on a user's reviews (needs `content`
                     and `score` columns; ~1 minute for ~100 reviews).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import reviewiq  # noqa: E402  (import after sys.path tweak)

FULL_RUN = ROOT / "data" / "processed" / "full_run"
REPORTS = ROOT / "reports"
APPS = ["chatgpt", "facebook", "netflix", "snapchat", "tiktok"]

st.set_page_config(page_title="ReviewIQ", page_icon="📱", layout="wide")
st.title("📱 ReviewIQ — Review Intelligence")
st.caption("Raw app reviews in → decision-ready product report out. "
           "No training, no APIs — two pretrained models and honest math.")

tab_explore, tab_upload = st.tabs(["📊 Explore the 5 apps", "⬆️ Upload your CSV"])

# ---------------------------------------------------------------- explore tab
with tab_explore:
    app = st.selectbox("App", APPS, format_func=str.title)

    report_file = REPORTS / f"{app}_report.md"
    topics_file = FULL_RUN / f"{app}_topics.parquet"
    clustered_file = FULL_RUN / f"{app}_clustered.parquet"

    if not report_file.exists():
        st.warning(f"No precomputed report for {app}. Run `python src/reviewiq.py` first.")
    else:
        left, right = st.columns([3, 2])

        with left:
            st.markdown(report_file.read_text(encoding="utf-8"))

        with right:
            df = pd.read_parquet(clustered_file, columns=["score", "sentiment", "cluster"])

            st.subheader("Sentiment (from review text)")
            st.bar_chart(df["sentiment"].value_counts())

            st.subheader("Star ratings")
            st.bar_chart(df["score"].value_counts().sort_index())

            n_topics = df.loc[df["cluster"] != -1, "cluster"].nunique()
            noise = (df["cluster"] == -1).mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews", f"{len(df):,}")
            c2.metric("Topics found", n_topics)
            c3.metric("Generic filler", f"{noise:.0%}")

        st.subheader("All topics")
        topics = pd.read_parquet(topics_file)
        st.dataframe(
            topics[["cluster", "n_reviews", "avg_score", "top_words", "example"]],
            use_container_width=True, hide_index=True,
        )

# ----------------------------------------------------------------- upload tab
with tab_upload:
    st.markdown(
        "Upload a CSV with at least a **`content`** column (review text) and a "
        "**`score`** column (1–5 stars). The full pipeline runs live: embeddings "
        "→ topic discovery → sentiment → report."
    )
    uploaded = st.file_uploader("Reviews CSV", type="csv")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        missing = {"content", "score"} - set(raw.columns)
        if missing:
            st.error(f"CSV is missing required column(s): {', '.join(missing)}")
        else:
            st.write(f"Loaded **{len(raw):,}** reviews.")
            if st.button("Run ReviewIQ", type="primary"):
                with st.spinner("Running the pipeline — embeddings, clustering, "
                                "sentiment, report… (~1 min per 100 reviews)"):
                    df, topics, report = reviewiq.run_pipeline(
                        raw.dropna(subset=["content"]), "your upload"
                    )
                st.success(f"Done — {len(df):,} reviews analyzed, "
                           f"{topics.shape[0]} topics found.")
                st.markdown(report)
                with st.expander("Topic table"):
                    st.dataframe(topics, use_container_width=True, hide_index=True)
                st.download_button("Download report (Markdown)", report,
                                   file_name="reviewiq_report.md")
