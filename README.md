# ScrutinizeIQ

**Turn thousands of app-store reviews into decision-ready product insights.**

ScrutinizeIQ is an NLP-powered review intelligence pipeline that transforms raw customer reviews into structured product insights: sentiment distribution, automatically discovered topics, ranked pain points, feature requests, positive themes, representative review quotes, and executive-style reports.

The project combines multilingual sentence embeddings, UMAP dimensionality reduction, HDBSCAN clustering, transformer-based sentiment analysis, and a reusable reporting pipeline. A Streamlit dashboard provides an interactive way to explore precomputed results and analyze new reviews.

> **Built as a collaborative project during the SPICED Academy Data Science bootcamp.**

---

## What ScrutinizeIQ Does

Product teams often have thousands of app-store reviews but limited time to manually understand them.

ScrutinizeIQ turns this:

```text
Thousands of raw customer reviews
                ↓
       Cleaning & validation
                ↓
       Semantic embeddings
                ↓
      Topic discovery & clustering
                ↓
       Sentiment classification
                ↓
       Product-level analysis
                ↓
     Decision-ready report
```

The system helps answer:

- What are customers complaining about most?
- Which problems receive the lowest ratings?
- What features are customers requesting?
- What do customers like about the product?
- Which topics contain mostly negative feedback?
- How is sentiment distributed across the review corpus?
- What representative customer quotes illustrate each issue?

## Key Features

### Multilingual NLP

Each review is converted into a semantic embedding using a pretrained multilingual sentence-transformer model.

### Automatic Topic Discovery

The pipeline uses UMAP for dimensionality reduction and HDBSCAN for density-based clustering. This allows the system to discover topic structure automatically while isolating generic or low-information reviews as noise.

### Sentiment Analysis

A pretrained multilingual transformer classifies reviews as negative, neutral, or positive.

### Decision-Ready Reports

The reporting layer combines topic frequencies, average ratings, sentiment, representative reviews, pain points, positive themes, feature requests, and executive-level summaries.

### Interactive Streamlit Dashboard

The dashboard supports:

- Exploring precomputed results for five applications
- Viewing sentiment and rating distributions
- Exploring discovered topics
- Uploading custom review CSV files
- Analyzing new reviews
- Downloading generated reports

---

## Validation Highlights

### Sentiment

On the documented validation comparison:

- Rule-based VADER identified approximately **51%** of 1-star negative reviews.
- The multilingual transformer identified approximately **73%** at comparable positive accuracy of approximately **87%**.

### Topic Clustering

A baseline comparison demonstrated the effect of dimensionality reduction:

| Approach | Silhouette Score |
|---|---:|
| KMeans on raw 384-dimensional embeddings | 0.035 |
| KMeans after UMAP | 0.34 |

HDBSCAN builds on the reduced representation while automatically discovering topic structure and isolating generic filler reviews as noise.

### Development Corpus

The project was developed and evaluated using approximately **496,000 Google Play reviews** covering:

- ChatGPT
- Facebook
- Netflix
- Snapchat
- TikTok

The datasets were sourced from public Kaggle datasets.

## Example Output

The repository includes generated reports under reports/.

For example:

- [Netflix report](reports/netflix_report.md)
- [ChatGPT report](reports/chatgpt_report.md)
- [Facebook report](reports/facebook_report.md)
- [Snapchat report](reports/snapchat_report.md)
- [TikTok report](reports/tiktok_report.md)

The reports contain executive summaries, sentiment breakdowns, ranked pain points, average ratings by topic, representative customer quotes, and discovered topics.



---

## How It Works

```text
                         RAW REVIEWS
                              │
                              ▼
                    Cleaning & Validation
                    ├── deduplication
                    ├── missing-value handling
                    └── language processing
                              │
                              ▼
                    Semantic Embeddings
                  multilingual transformer
                              │
                              ▼
                       UMAP Reduction
                              │
                              ▼
                       HDBSCAN Clustering
                    ├── topic discovery
                    └── noise detection
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Sentiment Analysis          Topic Analysis
                 │                         │
                 └────────────┬────────────┘
                              ▼
                     Product Intelligence
                              │
                              ▼
                       Report Generator
                              │
                              ▼
                  Decision-Ready Insights
```

Where applicable, the pipeline uses fixed random seeds to improve reproducibility.

## Streamlit Dashboard

Launch the application from the repository root:

```powershell
streamlit run app.py
```

The dashboard provides two main workflows.

### Explore Apps

Explore precomputed analysis for:

- ChatGPT
- Facebook
- Netflix
- Snapchat
- TikTok

### Upload Reviews

Upload a CSV containing:

```text
content
score
```

The application then runs the analysis pipeline on the uploaded reviews and generates a report.

The dashboard also supports fetching recent reviews from Google Play and Apple App Store URLs where supported by the application.

---

## Repository Structure

```text
ScrutinizeIQ/
│
├── app.py
├── demo_upload.csv
├── README.md
│
├── src/
│   └── scrutinizeiq.py
│
├── notebooks/
│   ├── 01.EDA.ipynb
│   ├── 02.EDA.ipynb
│   ├── 03.baselines.ipynb
│   ├── 04.pipeline.ipynb
│   └── 05.train_sentiment.ipynb
│
├── reports/
│   ├── chatgpt_report.md
│   ├── facebook_report.md
│   ├── netflix_report.md
│   ├── netflix_10k_report.md
│   ├── snapchat_report.md
│   └── tiktok_report.md
│
├── docs/
│   ├── FINAL_DECK_UPDATES.md
│   ├── FULL_DECK_CONTENT.md
│   ├── HANDOFF.md
│   ├── LEARNING_NOTES.md
│   ├── PRESENTATION_QA.md
│   ├── SLIDE_FIXES.md
│   └── TECH_SLIDES_CONTENT.md
│
└── .streamlit/
    └── config.toml
```

Large raw datasets are intentionally excluded from the repository.

## Installation

### Requirements

- Python 3.13
- An environment capable of installing the required NLP and data-science packages

Install the main dependencies with:

```powershell
pip install pandas pyarrow matplotlib scikit-learn sentence-transformers hdbscan umap-learn langdetect nltk streamlit google-play-scraper
```

> `pyarrow` is required because the pipeline uses Parquet files for intermediate and processed data.

## Running the Dashboard

From the repository root:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

## Running the Pipeline

The reusable pipeline can be executed with:

```powershell
python src/scrutinizeiq.py
```

Pretrained models are downloaded automatically on first use and cached locally.

For large datasets, full processing can require several hours on CPU.

---

## Collaboration & Project Background

ScrutinizeIQ originated as **ReviewIQ**, a collaborative project developed with **Nithin Pranav** during the SPICED Academy Data Science bootcamp.

The project was developed collaboratively through live coding sessions, technical discussions, experimentation, debugging, and joint review of the pipeline and results.

This repository is my maintained portfolio version of the project, presented under the name **ScrutinizeIQ**.

The original collaborative Git history is intentionally preserved rather than rewritten.

## Project Development Approach

The project demonstrates an end-to-end data science workflow:

1. Data ingestion
2. Data cleaning
3. Exploratory data analysis
4. Language processing
5. Semantic representation
6. Dimensionality reduction
7. Unsupervised topic discovery
8. Sentiment analysis
9. Baseline comparison
10. Product-level interpretation
11. Automated report generation
12. Interactive application development

## Known Limitations

The project documents its limitations rather than hiding them.

- Romanized Hindi (Hinglish) can be incorrectly detected as English.
- Sentiment classification is not perfect and achieves approximately 73% detection of 1-star negative reviews in the documented comparison.
- Very short reviews often contain insufficient semantic information for meaningful topic clustering.
- App-store review availability and external store APIs/feeds can change over time.
- Large-scale processing can require substantial CPU time and memory.

## Future Improvements

Potential extensions include:

- Improved multilingual language detection
- Better handling of Romanized languages
- Aspect-based sentiment analysis
- Temporal topic tracking
- Product/version-level trend analysis
- Interactive topic exploration
- Automated alerts for emerging negative themes
- Richer dashboard visualizations
- Deployment as a hosted web application

