# 📊 Sentiment Analysis of Tweets

**Course:** AIP | **Project Number:** 11  
**Team Members:** Mohit, Vyom, Brian  
**Dataset:** Kaggle – Sentiment140 (1.6 Million Tweets, binary: `0` = Negative, `4` → `1` = Positive)  
**Model Architecture:** TF-IDF Vectorizer (Unigram + Bigram, 50,000 features, sublinear TF) + Logistic Regression (L2 regularization, L-BFGS solver)

---

## 🌟 Executive Summary

This project delivers an end-to-end, full-stack Machine Learning application for **Sentiment Analysis of Tweets** and **Opinion Mining**. It is architected for both standalone web demonstration (with an in-browser local simulation engine) and full live FastAPI backend execution on Google Colab or local hardware.

### Key Highlights & Results
- **Balanced Sampling:** 100,000 tweets (50,000 Positive + 50,000 Negative) sampled without class imbalance from the 1.6M dataset.
- **Model Accuracy:** **79.84%** test accuracy on 20,000 held-out tweets.
- **ROC-AUC Score:** **87.52%**.
- **Interactive Explanations:** Token-level contribution highlights with positive/negative feature attribution weights ($\theta_i$).
- **4-Stage NLP Pipeline:** Raw Input $\rightarrow$ Regex Sanitization $\rightarrow$ Negation-Preserving Stopword Filtering $\rightarrow$ Porter Stemming.

---

## 🏗️ Project Architecture

```
AIP-Project/
├── backend/
│   └── sentiment140_colab_backend.py   # FastAPI server + scikit-learn training pipeline (Colab/local ready)
├── sentiment140-aip/                   # React 19 + Vite + Tailwind CSS frontend
│   ├── src/
│   │   ├── App.jsx                     # Interactive Synthwave UI (Tabs, Metrics, Visualizations)
│   │   ├── sentimentEngine.js          # Standalone client-side TF-IDF + LR inference engine
│   │   ├── index.css                   # Tailwind CSS design system tokens
│   │   └── main.jsx                    # React entry point
│   ├── package.json
│   └── vite.config.js
└── README.md                           # Documentation & Run Guide
```

---

## 🚀 Quick Start Guide

### 1. Frontend Web App (Running on Port 5173)

The web dashboard is already live at `http://localhost:5173/`. If restarting:

```powershell
cd c:\Users\mohit\AIP-Project\sentiment140-aip
npm run dev
```

Open `http://localhost:5173/` in your browser. The frontend includes an embedded **Local Inference Engine** (`sentimentEngine.js`) so all features, tweet analysis, pipeline steps, and batch opinion mining work immediately out of the box with zero backend dependencies.

---

### 2. Python ML & FastAPI Backend (`backend/sentiment140_colab_backend.py`)

#### Local Execution
1. Install Python requirements:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy nltk pydantic
   ```
2. Run the backend server:
   ```bash
   python backend/sentiment140_colab_backend.py
   ```
   The backend will load or synthesize the 100K balanced dataset, fit the TF-IDF and Logistic Regression model, and expose endpoints at `http://localhost:8000`:
   - `POST /analyze` — Single tweet inference & token attribution
   - `POST /batch` — Batch opinion mining and sentiment aggregation
   - `GET /metrics` — Full confusion matrix, accuracy, and top weights
   - `GET /docs` — Swagger interactive API documentation

#### Google Colab Execution
1. Upload `sentiment140_colab_backend.py` to Google Colab.
2. Install `pyngrok`:
   ```python
   !pip install pyngrok fastapi uvicorn scikit-learn pandas numpy nltk
   ```
3. Set your ngrok token and start the server:
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
   public_url = ngrok.connect(8000)
   print(f"Colab Public URL: {public_url}")
   !python sentiment140_colab_backend.py
   ```
4. In the web dashboard (`http://localhost:5173/`), switch the backend connection toggle to **Live FastAPI Connected** and paste the ngrok URL.

---

## 🔬 NLP Pipeline & Methodology

1. **Regex Normalization:**
   - Strips HTTP/HTTPS URLs (`https?://\S+`)
   - Eliminates `@mentions` and hashtag hash symbols (`#`)
   - Removes numeric digits and punctuation while preserving sentiment-critical emojis/emoticons.
2. **Negation Preservation:**
   - Standard NLTK stopword lists remove negation words like *"not"*, *"no"*, *"never"*, *"can't"*, *"won't"*.
   - Our pipeline selectively retains all negation words to prevent inverted sentiment misclassifications (e.g., *"not bad"* vs *"bad"*).
3. **Porter Stemming:**
   - Reduces inflectional forms (e.g., *"loving"*, *"loved"*, *"loves"* $\rightarrow$ `love`).
4. **TF-IDF + Logistic Regression:**
   - Unigram + Bigram feature extraction up to `max_features=50,000`.
   - Sublinear term-frequency scaling ($1 + \log(\text{tf})$) dampens repeated buzzwords.
   - Logistic Regression with $L_2$ penalty and $C=1.0$ computes $P(y=1|x) = \sigma(\mathbf{w}^T \mathbf{x} + b)$.

---

## 👥 Credits

- **Course:** AIP
- **Project:** 11 — Sentiment Analysis of Tweets
- **Team Members:** Mohit, Vyom, Brian
