# Quora Question Pair Duplicate Detection — BERT Models

A comparative study of modern transformer-based approaches for detecting duplicate questions on the Quora platform. This repository contains two notebooks implementing and comparing **4 models** across **3 fundamentally different NLP paradigms**.

---

## 📁 Notebooks

| Notebook | Models Inside |
|---|---|
| `deberta_crossencoder.ipynb` | DeBERTa-v3 Cross-Encoder |
| `sbert_setfit_stacking.ipynb` | SBERT Bi-Encoder · RoBERTa Cross-Encoder · Stacking Ensemble |

---

## 📊 Dataset

**Quora Question Pairs** — [Kaggle](https://www.kaggle.com/c/quora-question-pairs)

| Property | Value |
|---|---|
| Full dataset | 400,000+ question pairs |
| Sample used | 30,000 pairs |
| Train / Val / Test | 72% / 8% / 20% |
| Class balance | 63% Not Duplicate / 37% Duplicate |
| Split seed | `random_state=42` (identical across all notebooks for fair comparison) |

---

## 🧠 Models

### 1. DeBERTa-v3 Cross-Encoder
**Notebook:** `deberta_crossencoder.ipynb`

**Architecture:** Both questions fed together as one sequence — `[CLS] Q1 [SEP] Q2 [SEP]`. Every token in Q1 directly attends to every token in Q2 at every transformer layer via **disentangled attention** (position and content encoded separately).

**Model:** `microsoft/deberta-v3-small` (44M parameters)

**Why unique over BERT:** DeBERTa separates position and content into independent vectors. Standard BERT sums them together, losing the ability to reason about positional and semantic similarity independently. DeBERTa achieves better accuracy with 3× fewer parameters than RoBERTa.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8700 |
| F1 (weighted) | 0.8707 |
| AUC-ROC | 0.9387 |

**Outputs generated:**
- `deberta_results.png` — confusion matrix, loss curve, ROC curve

---

### 2. SBERT Bi-Encoder
**Notebook:** `sbert_setfit_stacking.ipynb`

**Architecture:** Q1 and Q2 encoded **separately** into 768-dim vectors using `sentence-transformers/all-mpnet-base-v2`. Classification uses interaction features: cosine similarity + element-wise difference + element-wise product → Logistic Regression.

**Why unique:** Bi-encoders allow pre-computing and caching all question embeddings. At inference time only one forward pass is needed per new question — 100× faster than cross-encoders at scale. This is how production search engines (Google, Bing) do semantic search.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8392 |
| F1 (weighted) | 0.8400 |
| AUC-ROC | 0.9153 |

**Outputs generated:**
- `sbert_evaluation.png` — confusion matrix, ROC curve, PR curve
- `sbert_cosine_analysis.png` — cosine similarity distribution by class + threshold sweep
- `shap_sbert_bar.png` — SHAP feature importance (cosine_similarity dominates)
- `shap_sbert_beeswarm.png` — direction of each feature's effect
- `shap_sbert_waterfall.png` — individual prediction explanation
- `sbert_error_analysis.png` — false positive / false negative breakdown

---

### 3. RoBERTa Cross-Encoder
**Notebook:** `sbert_setfit_stacking.ipynb`

**Architecture:** Same cross-encoder pattern as DeBERTa — `[CLS] Q1 [SEP] Q2 [SEP]` fed as one sequence. Uses `roberta-base` (125M parameters) fine-tuned end-to-end for binary classification.

**Why unique over BERT:** RoBERTa removes BERT's flawed Next Sentence Prediction objective and uses dynamic masking — strictly better pre-training. Every token in Q1 attends to every token in Q2 directly, enabling richer cross-question interaction than bi-encoders.

**Training:**

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Max sequence length | 128 tokens |
| Best epoch | Epoch 2 (val F1 = 0.8537) |

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8640 |
| F1 (weighted) | 0.8650 |
| AUC-ROC | 0.9373 |

**Outputs generated:**
- `roberta_evaluation.png` — confusion matrix, ROC curve, PR curve
- `roberta_comparison.png` — training curve + all models bar chart
- `shap_roberta_bar.png` — top 20 most influential words
- `roberta_error_analysis.png` — false positive / false negative breakdown

---

### 4. Stacking Ensemble
**Notebook:** `sbert_setfit_stacking.ipynb`

**Architecture:** Meta-learning approach. Takes probability outputs from SBERT and RoBERTa plus hand-crafted features, feeds them into a Logistic Regression meta-classifier that learns when to trust which model.

**Meta-classifier inputs:**

| Feature | Source |
|---|---|
| SBERT probability | `sbert_clf.predict_proba()[:,1]` |
| RoBERTa probability | `softmax(roberta_logits)[:,1]` |
| jaccard | word set intersection/union |
| word_share | common/total words |
| q1_len, q2_len | character counts |
| len_diff | absolute length difference |
| q1_words, q2_words | word counts |

**Why it works:** Each model makes different errors. SBERT fails on zero-vocabulary-overlap paraphrases. RoBERTa sometimes over-relies on surface patterns. Hand features catch simple cases. The meta-classifier covers each model's blind spots — SHAP analysis confirms it assigns highest weight to neural model probabilities.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8707 |
| F1 (weighted) | 0.8711 |
| AUC-ROC | **0.9426** ← best of all models |

**Outputs generated:**
- `stacking_evaluation.png` — all 3 models ROC curves on one plot
- `stacking_analysis.png` — meta-classifier coefficients + prediction breakdown
- `shap_stacking_bar.png` — which model the ensemble trusts most
- `shap_stacking_beeswarm.png` — SHAP direction of effects
- `shap_stacking_waterfall.png` — individual prediction explanation

---

## 📈 Full Results Comparison

| Model | Paradigm | Accuracy | F1 | AUC | Params |
|---|---|---|---|---|---|
| SBERT Bi-Encoder | Bi-encoder | 0.8392 | 0.8400 | 0.9153 | 109M |
| RoBERTa Cross-Encoder | Cross-encoder | 0.8640 | 0.8650 | 0.9373 | 125M |
| DeBERTa-v3 Cross-Encoder | Cross-encoder | 0.8700 | 0.8707 | 0.9387 | 44M |
| **Stacking Ensemble** | **Meta-learning** | **0.8707** | **0.8711** | **0.9426** | — |

**Key findings:**
- Cross-encoders (RoBERTa, DeBERTa) consistently outperform the bi-encoder (SBERT) by 2–3% — joint encoding of question pairs captures richer semantic interaction
- DeBERTa achieves best single-model accuracy with 3× fewer parameters than RoBERTa — architectural innovation (disentangled attention) matters more than model size
- Stacking ensemble achieves best AUC (0.9426) by combining diverse model signals
- SHAP analysis confirms cosine similarity is the single most predictive feature in the SBERT classifier out of 1,537 total features

---

## 🗂️ Notebook Structure

### `deberta_crossencoder.ipynb`

| Cell | Purpose |
|---|---|
| Cell 1 | Install dependencies (`transformers==4.40.0`, `datasets`, `sentencepiece`) |
| Cell 2 | Imports |
| Cell 3 | Load + sample 30k rows, stratified train/val/test split |
| Cell 4 | Load DeBERTa tokenizer, verify cross-encoder input format |
| Cell 5 | `QuoraCrossEncoderDataset` class — tokenizes Q1+Q2 together |
| Cell 6 | Load `deberta-v3-small`, print parameter counts |
| Cell 7 | Training setup — AdamW, linear warmup scheduler, CrossEntropyLoss |
| Cell 8 | Training loop with best model checkpoint saving |
| Cell 9 | Load best checkpoint, evaluate on test set |
| Cell 10 | Loss curve, accuracy curve, confusion matrix plots |
| Cell 11 | Live inference function |

### `sbert_setfit_stacking.ipynb`

| Cell | Purpose |
|---|---|
| Cell 1–2 | Install + imports |
| Cell 3 | Load data (same split as DeBERTa — `random_state=42`) |
| Cell 4–5 | SBERT: encode all pairs, build interaction features, train LR classifier |
| Report Cells 1–5 | SBERT evaluation plots, cosine analysis, SHAP, error analysis |
| Cell 11 | Redefine `QuoraCrossEncoderDataset` (needed since different notebook) |
| Cell 12 | RoBERTa: load model, train 3 epochs, evaluate |
| Report Cells 1–4 | RoBERTa evaluation plots, training curve, SHAP, error analysis |
| Cell 17 | Stacking ensemble: hand features + meta-classifier training |
| Cell 18–21 | Stacking evaluation plots, SHAP cells |

---

## ⚙️ Requirements

```
torch
transformers==4.40.0
sentence-transformers
datasets
sentencepiece
protobuf
scikit-learn
pandas
numpy
matplotlib
seaborn
shap
```

Install all:
```bash
pip install transformers==4.40.0 sentence-transformers datasets sentencepiece protobuf shap scikit-learn
```

---

## 🚀 How to Run

### On Kaggle (recommended — free GPU)

1. Create a new notebook on [kaggle.com](https://www.kaggle.com)
2. Add the **Quora Question Pairs** dataset: `+ Add Data → Search "quora question pairs"`
3. Set accelerator: `Settings → Accelerator → GPU T4 x1`
4. Upload the notebook file
5. Click **Run All**

### Data path
```python
# Kaggle path used in both notebooks
df = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
```

### Run order
Run `deberta_crossencoder.ipynb` first (standalone).  
Then run `sbert_setfit_stacking.ipynb` (also standalone — no dependency on DeBERTa notebook).

---

## 🔍 Explainability (SHAP)

Both notebooks include SHAP analysis:

- **SBERT:** `LinearExplainer` on Logistic Regression — shows cosine_similarity is the dominant feature
- **RoBERTa:** Text masker with word-level attribution — shows which words drive duplicate prediction
- **Stacking:** `LinearExplainer` on meta-classifier — shows which model the ensemble trusts most

All SHAP cells install `shap` automatically via `!pip install shap -q` and can be run immediately after model evaluation without retraining.

---

## ❌ Error Analysis

Both notebooks include systematic error analysis identifying two failure modes:

**False Positives** (said duplicate, actually different):
- Pattern: same topic, different intent
- Example: *"What is Bitcoin?"* vs *"How do I buy Bitcoin?"*

**False Negatives** (said not duplicate, actually duplicate):
- Pattern: heavy paraphrase with zero shared vocabulary
- Example: *"How do I lose weight?"* vs *"What are ways to shed pounds?"*

---

