"""
ScrutinizeIQ dashboard — Streamlit UI over the pipeline in src/scrutinizeiq.py.

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
import scrutinizeiq  # noqa: E402  (import after sys.path tweak)

FULL_RUN = ROOT / "data" / "processed" / "full_run"
REPORTS = ROOT / "reports"
APPS = ["chatgpt", "facebook", "netflix", "snapchat", "tiktok"]

st.set_page_config(page_title="ScrutinizeIQ", page_icon="📱", layout="wide")
st.title("📱 ScrutinizeIQ — Review Intelligence")
st.caption("Raw app reviews in → decision-ready product report out. "
           "No training, no APIs — two pretrained models and honest math.")

tab_explore, tab_upload = st.tabs(["📊 Explore the 5 apps", "⬆️ Upload your CSV"])

# ---------------------------------------------------------------- explore tab
# ---------------------------------------------------------------- explore tab
with tab_explore:
    app = st.selectbox("App", APPS, format_func=str.title)

    report_file = REPORTS / f"{app}_report.md"
    topics_file = FULL_RUN / f"{app}_topics.parquet"
    clustered_file = FULL_RUN / f"{app}_clustered.parquet"

    if not report_file.exists():
        st.warning(
            f"No precomputed report for {app}. "
            "Run `python src/scrutinizeiq.py` first."
        )
    else:
        left, right = st.columns([3, 2])

        with left:
            st.markdown(report_file.read_text(encoding="utf-8"))

        with right:
            if clustered_file.exists():
                df = pd.read_parquet(
                    clustered_file,
                    columns=["score", "sentiment", "cluster"],
                )

                st.subheader("Sentiment (from review text)")
                st.bar_chart(df["sentiment"].value_counts())

                st.subheader("Star ratings")
                st.bar_chart(df["score"].value_counts().sort_index())

                n_topics = df.loc[
                    df["cluster"] != -1, "cluster"
                ].nunique()
                noise = (df["cluster"] == -1).mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Reviews", f"{len(df):,}")
                c2.metric("Topics found", n_topics)
                c3.metric("Generic filler", f"{noise:.0%}")
            else:
                st.info(
                    "Precomputed review data is not available locally. "
                    "The report below is available, but charts and topic "
                    "statistics require the processed Parquet files."
                )

        if topics_file.exists():
            st.subheader("All topics")
            topics = pd.read_parquet(topics_file)
            st.dataframe(
                topics[
                    ["cluster", "n_reviews", "avg_score", "top_words", "example"]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "The precomputed topic table is not available locally. "
                "The generated report is still available above."
            )

# ----------------------------------------------------------------- upload tab
def display_app_name(name: str) -> str:
    """Convert store identifiers into readable display names."""
    known_names = {
        "com.whatsapp": "WhatsApp",
        "com.facebook.katana": "Facebook",
        "com.netflix.mediaclient": "Netflix",
        "com.snapchat.android": "Snapchat",
        "com.zhiliaoapp.musically": "TikTok",
    }
    return known_names.get(name, name)


def report_filename(name: str) -> str:
    """Create a safe filename for a generated report."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{slug}_report.md"


def analyze_and_show(raw: pd.DataFrame, name: str):
    """Run the pipeline, render the results, and save the report."""
    display_name = display_app_name(name)

    with st.spinner(
        "Running the pipeline — embeddings, clustering, "
        "sentiment, report… (~1 min per 100 reviews)"
    ):
        df, topics, report = scrutinizeiq.run_pipeline(
            raw.dropna(subset=["content"]), display_name
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / report_filename(name)
    report_path.write_text(report, encoding="utf-8")

    st.success(
        f"Done — {len(df):,} reviews analyzed, "
        f"{topics.shape[0]} topics found."
    )

    st.markdown(report)
    st.caption(f"Report saved to `{report_path}`")

    with st.expander("Topic table"):
        st.dataframe(topics, width="stretch", hide_index=True)

    st.download_button(
        "Download report (Markdown)",
        report,
        file_name=report_filename(name),
    )

def play_store_app_id(url: str) -> str | None:
    """'https://play.google.com/store/apps/details?id=com.spotify.music&hl=en'
    -> 'com.spotify.music'. Also accepts a bare app id."""
    url = url.strip()
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    if url and "/" not in url and " " not in url:
        return url          # looks like a bare app id, e.g. com.spotify.music
    return None


def apple_store_app_info(url: str) -> tuple[str, str] | None:
    """'https://apps.apple.com/us/app/spotify-music/id324684580'
    -> ('324684580', 'us'). Returns None if it doesn't look like an Apple link."""
    import re
    m = re.search(r"apps\.apple\.com/(\w\w)/.*?/id(\d+)", url.strip())
    if m:
        return m.group(2), m.group(1)
    m = re.search(r"apps\.apple\.com/.*?/id(\d+)", url.strip())
    if m:
        return m.group(1), "us"
    return None


def fetch_apple_reviews(app_id: str, country: str, n: int) -> pd.DataFrame:
    """Fetch newest reviews from Apple's public RSS feed (50 per page,
    up to ~500). Returns a DataFrame with reviewId / content / score."""
    import json
    import urllib.request

    rows = []
    for page in range(1, min(10, (n + 49) // 50) + 1):
        url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
               f"page={page}/id={app_id}/sortby=mostrecent/json")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                entries = json.load(r)["feed"].get("entry", [])
        except Exception:
            break
        # the first entry on page 1 is app metadata, not a review
        for e in entries:
            if "im:rating" not in e:
                continue
            rows.append({
                "reviewId": e["id"]["label"],
                "content": (e["title"]["label"] + ". " + e["content"]["label"]).strip(),
                "score": int(e["im:rating"]["label"]),
            })
        if len(entries) < 50:
            break
    return pd.DataFrame(rows).head(n)


with tab_upload:
    st.subheader("Paste an app-store link")
    st.caption("Google Play or Apple App Store — we fetch the newest reviews live.")
    link = st.text_input("Play Store / App Store URL",
                         placeholder="https://play.google.com/store/apps/details?id=...  or  https://apps.apple.com/us/app/...")
    n_reviews = st.slider("How many recent reviews", 50, 500, 200, step=50)

    if st.button("Fetch & analyze", type="primary", disabled=not link):
        apple = apple_store_app_info(link)
        google_id = None if apple else play_store_app_id(link)

        if apple is None and google_id is None:
            st.error("That doesn't look like a Play Store or App Store link.")
        else:
            try:
                if apple:
                    app_id, country = apple
                    with st.spinner(f"Fetching newest App Store reviews (id {app_id}, {country})…"):
                        raw = fetch_apple_reviews(app_id, country, n_reviews)
                    source = f"App Store id {app_id}"
                else:
                    from google_play_scraper import reviews as gp_reviews, Sort
                    with st.spinner(f"Fetching {n_reviews} newest reviews for {google_id}…"):
                        fetched, _ = gp_reviews(google_id, lang="en", country="us",
                                                sort=Sort.NEWEST, count=n_reviews)
                    raw = pd.DataFrame(fetched)[["reviewId", "content", "score"]] if fetched else pd.DataFrame()
                    source = display_app_name(google_id)





                if raw.empty:
                    st.error("No reviews returned — check the link.")
                else:
                    st.write(f"Fetched **{len(raw):,}** live reviews for `{source}`.")
                    analyze_and_show(raw, source)
            except Exception as e:
                st.error(f"Fetch failed ({type(e).__name__}) — no internet or the store "
                         f"changed something. Use the CSV upload below instead.")

    st.divider()
    st.subheader("…or upload a CSV")
    st.caption("Needs a `content` column (review text) and a `score` column (1–5 stars).")
    uploaded = st.file_uploader("Reviews CSV", type="csv")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        missing = {"content", "score"} - set(raw.columns)
        if missing:
            st.error(f"CSV is missing required column(s): {', '.join(missing)}")
        else:
            st.write(f"Loaded **{len(raw):,}** reviews.")
            if st.button("Run ScrutinizeIQ on the CSV"):
                analyze_and_show(raw, "your upload")
