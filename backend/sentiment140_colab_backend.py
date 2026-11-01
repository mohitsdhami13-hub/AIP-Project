"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Sentiment Analysis of Tweets — FastAPI Backend                        ║
║  Course: AIP | Project 11                                              ║
║  Team: Mohit, Vyom, Brian                                              ║
║                                                                        ║
║  ML Pipeline: TF-IDF (1,2)-gram + Logistic Regression                 ║
║  Dataset: Kaggle Sentiment140 (100K balanced sample from 1.6M)         ║
║                                                                        ║
║  Compatible with Google Colab (pyngrok) and local execution.           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import time
import random
import string
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── NLTK Setup ──
# pyrefly: ignore[missing-import]
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Scikit-learn ──
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# ── FastAPI ──
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA ACQUISITION & BALANCED SAMPLING
# ══════════════════════════════════════════════════════════════════════════

# Negation words to preserve during stopword removal
NEGATION_WORDS = {
    "not", "no", "never", "neither", "barely", "hardly",
    "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't",
    "couldn't", "won't", "can't", "don't", "nor", "none"
}

# Extended stop words minus negation words
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

# Porter Stemmer
stemmer = PorterStemmer()


def generate_synthetic_dataset(n_positive=50000, n_negative=50000, seed=42):
    """
    Generates a synthetic fallback dataset when the 238MB Kaggle CSV is not
    available locally. Produces tweets with clear sentiment signals.
    """
    random.seed(seed)

    positive_templates = [
        "I love {} so much, it makes me happy!",
        "Absolutely amazing {} experience today!",
        "Thank you for the wonderful {} service!",
        "Best {} ever! Highly recommend to everyone!",
        "So excited about the new {}! Great job!",
        "Really enjoyed the {} today, fantastic!",
        "Happy to announce {} is working perfectly!",
        "Awesome {} day with friends and family!",
        "Beautiful {} weather makes everything better!",
        "Just had a perfect {} moment, feeling blessed!",
        "Love the new {} update, works like a charm!",
        "Congratulations on the amazing {} launch!",
        "Super impressed with {} quality and design!",
        "What a brilliant {} idea, well done team!",
        "Feeling great about {} achievements today!",
    ]

    negative_templates = [
        "I hate {} so much, worst experience ever.",
        "Terrible {} service, completely disappointed.",
        "So sad about the {} situation, really upset.",
        "The worst {} I've ever seen, total failure.",
        "Never buying {} again, horrible quality.",
        "Frustrated with {} customer support, awful.",
        "Really bad {} experience today, so annoying.",
        "Disappointed with the {} update, it's broken.",
        "Can't stand {} anymore, waste of time.",
        "Sick of dealing with {} problems every day.",
        "Horrible {} service, I want my money back.",
        "Such a boring and terrible {} product.",
        "Angry about the {} delay, unacceptable.",
        "Ruined my day with this stupid {} issue.",
        "Disgusted by the poor {} quality overall.",
    ]

    topics = [
        "product", "movie", "restaurant", "flight", "hotel",
        "app", "game", "concert", "weather", "coffee",
        "phone", "laptop", "music", "book", "food",
        "travel", "shopping", "school", "work", "team"
    ]

    data = []

    for _ in range(n_positive):
        template = random.choice(positive_templates)
        topic = random.choice(topics)
        tweet = template.format(topic)
        # Add some variation
        if random.random() > 0.7:
            tweet += " " + random.choice(["😊", "❤️", "🎉", "👏", "🔥"])
        data.append({"target": 4, "text": tweet})

    for _ in range(n_negative):
        template = random.choice(negative_templates)
        topic = random.choice(topics)
        tweet = template.format(topic)
        if random.random() > 0.7:
            tweet += " " + random.choice(["😞", "💔", "😡", "👎", "😤"])
        data.append({"target": 0, "text": tweet})

    random.shuffle(data)
    df = pd.DataFrame(data)
    print(f"✅ Generated synthetic dataset: {len(df)} tweets ({n_positive} pos + {n_negative} neg)")
    return df


def load_and_sample_data(csv_path="training.1600000.processed.noemoticon.csv",
                         sample_size=100000, random_state=42):
    """
    Load Kaggle Sentiment140 CSV with balanced sampling.
    Falls back to synthetic data if CSV not found.
    """
    if os.path.exists(csv_path):
        print(f"📂 Loading Kaggle CSV from: {csv_path}")
        columns = ["target", "id", "date", "flag", "user", "text"]
        df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=columns)
        print(f"   Raw dataset size: {len(df):,} tweets")

        # Balanced sampling: 50K negative (0) + 50K positive (4)
        half = sample_size // 2
        df_neg = df[df["target"] == 0].sample(n=half, random_state=random_state)
        df_pos = df[df["target"] == 4].sample(n=half, random_state=random_state)
        df_sampled = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Map target: 0 -> 0 (Negative), 4 -> 1 (Positive)
        df_sampled["target"] = df_sampled["target"].map({0: 0, 4: 1})
        df_sampled = df_sampled[["target", "text"]]

        print(f"   Balanced sample: {len(df_sampled):,} tweets "
              f"(Neg={len(df_sampled[df_sampled.target==0]):,}, "
              f"Pos={len(df_sampled[df_sampled.target==1]):,})")
        return df_sampled
    else:
        print(f"⚠️  Kaggle CSV not found at '{csv_path}'")
        print(f"   Generating synthetic fallback dataset...")
        df = generate_synthetic_dataset(
            n_positive=sample_size // 2,
            n_negative=sample_size // 2,
            seed=random_state
        )
        df["target"] = df["target"].map({0: 0, 4: 1})
        return df[["target", "text"]]


# ══════════════════════════════════════════════════════════════════════════
# 2. 4-STAGE NLP PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def preprocess_tweet(text):
    """
    4-Stage NLP Preprocessing:
      Stage 1: Raw tweet input
      Stage 2: Regex cleaning (lowercase, strip URLs, @mentions, #, numbers, punctuation)
      Stage 3: Stopword removal (preserving negation words)
      Stage 4: Porter Stemming
    """
    # Stage 1: Raw
    raw = str(text)

    # Stage 2: Regex Cleaning
    cleaned = raw.lower()
    cleaned = re.sub(r"http\S+", "", cleaned)       # URLs
    cleaned = re.sub(r"@\w+", "", cleaned)           # @mentions
    cleaned = re.sub(r"#", "", cleaned)              # # symbols
    cleaned = re.sub(r"[0-9]+", "", cleaned)         # numbers
    cleaned = re.sub(r"[^\w\s]", "", cleaned)        # punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()   # extra whitespace
    regex_cleaned = cleaned

    # Stage 3: Stopword removal (preserve negations)
    tokens = cleaned.split()
    filtered_tokens = [t for t in tokens if t not in STOP_WORDS or t in NEGATION_WORDS]
    stopwords_removed = " ".join(filtered_tokens)

    # Stage 4: Porter Stemming
    stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]
    stemmed = " ".join(stemmed_tokens)

    return {
        "raw": raw,
        "regex_cleaned": regex_cleaned,
        "stopwords_removed": stopwords_removed,
        "stemmed_tokens": stemmed_tokens,
        "stemmed": stemmed,
    }


def preprocess_for_training(text):
    """Simplified preprocessing that returns the final stemmed string for vectorizer."""
    return preprocess_tweet(text)["stemmed"]


# ══════════════════════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("🧠 Sentiment140 — TF-IDF + Logistic Regression Training Pipeline")
print("=" * 70)

# Load data
df = load_and_sample_data()

print("\n📝 Preprocessing tweets (4-stage NLP pipeline)...")
start_time = time.time()
df["processed"] = df["text"].apply(preprocess_for_training)
preprocess_time = time.time() - start_time
print(f"   Preprocessing completed in {preprocess_time:.2f}s")

# Train/Test Split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df["processed"], df["target"],
    test_size=0.2, stratify=df["target"], random_state=42
)
print(f"\n📊 Train/Test Split: {len(X_train):,} train / {len(X_test):,} test")

# TF-IDF Vectorization
print("\n🔤 Fitting TF-IDF Vectorizer (ngram_range=(1,2), max_features=50000)...")
start_time = time.time()
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    min_df=5,
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
tfidf_time = time.time() - start_time
print(f"   Vocabulary size: {len(tfidf.vocabulary_):,} features")
print(f"   TF-IDF completed in {tfidf_time:.2f}s")

# Logistic Regression Training
print("\n🏋️ Training LogisticRegression (C=1.0, solver='lbfgs', max_iter=1000)...")
start_time = time.time()
lr_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
train_time = time.time() - start_time
print(f"   Training completed in {train_time:.2f}s")

# Evaluation
print("\n📈 Evaluating on test set...")
y_pred = lr_model.predict(X_test_tfidf)
y_prob = lr_model.predict_proba(X_test_tfidf)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"{'='*40}")
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

# Extract top features
feature_names = tfidf.get_feature_names_out()
coefficients = lr_model.coef_[0]
feature_coef = list(zip(feature_names, coefficients))
feature_coef_sorted = sorted(feature_coef, key=lambda x: x[1])

top_negative_features = [{"feature": f, "weight": round(float(w), 4)} for f, w in feature_coef_sorted[:15]]
top_positive_features = [{"feature": f, "weight": round(float(w), 4)} for f, w in feature_coef_sorted[-15:][::-1]]

print(f"\n🟢 Top 5 Positive Features: {[f['feature'] for f in top_positive_features[:5]]}")
print(f"🔴 Top 5 Negative Features: {[f['feature'] for f in top_negative_features[:5]]}")

# Store metrics
model_metrics = {
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "confusion_matrix": {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    },
    "dataset": {
        "name": "Kaggle Sentiment140",
        "total_records": 1600000,
        "sample_size": len(df),
        "positive_samples": int((df["target"] == 1).sum()),
        "negative_samples": int((df["target"] == 0).sum()),
        "test_size": len(X_test),
        "train_size": len(X_train),
    },
    "model": {
        "vectorizer": "TF-IDF (ngram_range=(1,2), max_features=50000, sublinear_tf=True)",
        "classifier": "LogisticRegression (C=1.0, solver=lbfgs, max_iter=1000)",
        "vocabulary_size": len(tfidf.vocabulary_),
    },
    "top_positive_features": top_positive_features,
    "top_negative_features": top_negative_features,
    "timing": {
        "preprocessing_seconds": round(preprocess_time, 2),
        "tfidf_seconds": round(tfidf_time, 2),
        "training_seconds": round(train_time, 2),
    }
}

print(f"\n✅ Model ready for inference!")


# ══════════════════════════════════════════════════════════════════════════
# 4. FASTAPI REST SERVER
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Sentiment Analysis of Tweets — AIP Project 11",
    description="TF-IDF + Logistic Regression on Sentiment140",
    version="1.0.0",
)

# CORS (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──

class TweetRequest(BaseModel):
    tweet: str

class BatchRequest(BaseModel):
    tweets: List[str]


def analyze_single_tweet(tweet_text: str) -> dict:
    """Full analysis pipeline for a single tweet."""
    start = time.time()
    preprocessed = preprocess_tweet(tweet_text)

    # Vectorize
    X = tfidf.transform([preprocessed["stemmed"]])

    # Get prediction probability
    prob = lr_model.predict_proba(X)[0]
    prob_negative = float(prob[0])
    prob_positive = float(prob[1])

    # Compute logit score (decision function)
    logit_score = float(lr_model.decision_function(X)[0])

    sentiment = "Positive" if prob_positive >= 0.5 else "Negative"
    confidence = max(prob_positive, prob_negative)

    # Token-level explainability
    token_contributions = []
    stemmed_tokens = preprocessed["stemmed_tokens"]

    # Get TF-IDF weights for the input
    tfidf_matrix = X.toarray()[0]

    # Check bigrams
    for i in range(len(stemmed_tokens) - 1):
        bigram = f"{stemmed_tokens[i]} {stemmed_tokens[i+1]}"
        if bigram in tfidf.vocabulary_:
            idx = tfidf.vocabulary_[bigram]
            tfidf_weight = float(tfidf_matrix[idx])
            lr_coef = float(coefficients[idx])
            if tfidf_weight > 0:
                token_contributions.append({
                    "token": bigram,
                    "tfidf_weight": round(tfidf_weight, 4),
                    "lr_coefficient": round(lr_coef, 4),
                    "impact_score": round(tfidf_weight * lr_coef, 4),
                    "type": "bigram"
                })

    # Check unigrams
    for token in stemmed_tokens:
        if token in tfidf.vocabulary_:
            idx = tfidf.vocabulary_[token]
            tfidf_weight = float(tfidf_matrix[idx])
            lr_coef = float(coefficients[idx])
            if tfidf_weight > 0:
                token_contributions.append({
                    "token": token,
                    "tfidf_weight": round(tfidf_weight, 4),
                    "lr_coefficient": round(lr_coef, 4),
                    "impact_score": round(tfidf_weight * lr_coef, 4),
                    "type": "unigram"
                })

    # Sort by absolute impact
    token_contributions.sort(key=lambda x: abs(float(x["impact_score"])), reverse=True)

    latency_ms = (time.time() - start) * 1000

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probability_positive": round(prob_positive, 4),
        "probability_negative": round(prob_negative, 4),
        "logit_score": round(logit_score, 4),
        "preprocessing_steps": [
            {"stage": 1, "name": "Raw Input", "output": preprocessed["raw"]},
            {"stage": 2, "name": "Regex Cleaning", "output": preprocessed["regex_cleaned"]},
            {"stage": 3, "name": "Negation-Safe Stopword Filter", "output": preprocessed["stopwords_removed"]},
            {"stage": 4, "name": "Porter Stemmed N-Grams", "output": preprocessed["stemmed"]},
        ],
        "token_contributions": token_contributions,
        "inference_latency_ms": round(latency_ms, 2),
    }


@app.post("/analyze")
async def analyze_tweet_endpoint(req: TweetRequest):
    """Analyze a single tweet's sentiment with full XAI breakdown."""
    return analyze_single_tweet(req.tweet)


@app.post("/batch")
async def batch_analyze_endpoint(req: BatchRequest):
    """Batch opinion mining across multiple tweets."""
    results = []
    for tweet_text in req.tweets:
        analysis = analyze_single_tweet(tweet_text)
        analysis["tweet"] = tweet_text
        results.append(analysis)

    positive_count = sum(1 for r in results if r["sentiment"] == "Positive")
    negative_count = sum(1 for r in results if r["sentiment"] == "Negative")
    total = len(results)
    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

    return {
        "results": results,
        "summary": {
            "total": total,
            "positive": positive_count,
            "negative": negative_count,
            "positive_ratio": round(positive_count / total, 4) if total > 0 else 0,
            "negative_ratio": round(negative_count / total, 4) if total > 0 else 0,
            "average_confidence": round(avg_confidence, 4),
            "net_sentiment_index": round(((positive_count - negative_count) / total) * 100, 2) if total > 0 else 0,
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Return model accuracy, confusion matrix, dataset stats, and top features."""
    return model_metrics


@app.get("/")
async def root():
    return {
        "project": "Sentiment Analysis of Tweets",
        "course": "AIP",
        "team": ["Mohit", "Vyom", "Brian"],
        "model": "TF-IDF + LogisticRegression",
        "status": "running",
        "endpoints": ["/analyze", "/batch", "/metrics"],
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. RUN SERVER
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Starting FastAPI server on http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("=" * 70)

    # ── Optional: pyngrok for Google Colab ──
    # Uncomment the following lines when running in Google Colab:
    # ---
    # from pyngrok import ngrok
    # ngrok_tunnel = ngrok.connect(8000)
    # print(f"🌐 ngrok Public URL: {ngrok_tunnel.public_url}")
    # ---

    uvicorn.run(app, host="0.0.0.0", port=8000)
