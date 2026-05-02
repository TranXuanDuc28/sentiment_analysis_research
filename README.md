# Cross-Domain & Cross-Lingual Sentiment Analysis with XLM-RoBERTa

A complete end-to-end PyTorch project that fine-tunes **XLM-RoBERTa** on
Amazon Reviews Multi (MARC) and evaluates it across **5 experimental scenarios**
covering in-domain, domain shift, language shift, double shift, and
cross-dataset generalisation.

---

## Project Structure

```
project/
├── data/
│   └── vsfc/              ← (optional) local UIT-VSFC files
│       ├── train/
│       │   ├── sents.txt
│       │   └── sentiments.txt
│       ├── dev/
│       └── test/
├── src/
│   ├── __init__.py
│   ├── dataset.py         ← Dataset loaders + PyTorch Dataset/DataLoader
│   ├── train.py           ← Model builder + training loop
│   ├── evaluate.py        ← Inference + metrics + visualisation
│   └── utils.py           ← Label maps, device setup, helpers
├── checkpoints/           ← Saved model checkpoint (auto-created)
├── results/               ← Plots + JSON results (auto-created)
├── main.py                ← Entry point — runs all 5 scenarios
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Language | Domain | Source |
|---------|----------|--------|--------|
| Amazon Reviews Multi (MARC) | English (+ Vietnamese) | Books, Electronics | [HuggingFace](https://huggingface.co/datasets/amazon_reviews_multi) |
| UIT-VSFC | Vietnamese | Student feedback | [GitHub](https://github.com/nguyenlab-sfl/UIT-VSFC) |

### Label Mapping

| Rating / Original | Sentiment Label |
|-------------------|-----------------|
| 1–2 stars (MARC) / 0 (VSFC) | 0 — **Negative** |
| 3 stars (MARC) / 1 (VSFC) | 1 — **Neutral** |
| 4–5 stars (MARC) / 2 (VSFC) | 2 — **Positive** |

---

## Experimental Scenarios

| # | Name | Train | Test |
|---|------|-------|------|
| 1 | In-domain | EN Books | EN Books |
| 2 | Domain Shift | EN Books | EN Electronics |
| 3 | Language Shift | EN Books | VI MARC (mixed) |
| 4 | Double Shift | EN Books | VI Electronics (MARC) |
| 5 | Cross-Dataset | Amazon EN | UIT-VSFC VI |

---

## Quick Start

### 1. Install dependencies

```bash
# (Recommended) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run in **quick mode** (smoke test, ~5–10 min on CPU)

```bash
cd project
python main.py --quick
```

### 3. Run **full experiment** (GPU recommended, ~2–4 h)

```bash
cd project
python main.py
```

> **Note:** The first run downloads `xlm-roberta-base` (~1.1 GB) and the
> MARC dataset from HuggingFace Hub. Both are cached automatically.

### 4. Re-use a saved checkpoint

If `checkpoints/xlm-roberta-books-en/config.json` already exists,
`main.py` loads that checkpoint automatically and **skips training**.
Delete the `checkpoints/` folder to force a fresh fine-tune.

---

## Outputs

After a successful run, you will find:

```
results/
├── all_results.json         ← Accuracy + F1 for all 5 scenarios
├── comparison.png           ← Side-by-side bar chart
└── confusion_matrices.png   ← Grid of confusion matrix heatmaps
```

### Example console output

```
============================================================
  FULL RESULTS SUMMARY
============================================================
Scenario                            Accuracy   F1-Macro  F1-Weighted
---------------------------------------------------------------------------
1. In-domain (EN Books → EN Books)    0.8340     0.8120       0.8290
2. Domain Shift (EN Books → EN ...)   0.7980     0.7750       0.7920
3. Language Shift (EN Books → VI ..)  0.6120     0.5830       0.5970
4. Double Shift (EN Books → VI El..)  0.5870     0.5540       0.5700
5. Cross-Dataset (Amazon EN → VSFC)   0.5430     0.4910       0.5100
---------------------------------------------------------------------------
```

*(Exact numbers will vary by hardware, random seed, and sample sizes.)*

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Base model | `xlm-roberta-base` |
| Classification head | Linear(768 → 3) |
| Max sequence length | 128 (quick: 64) |
| Optimiser | AdamW (lr=2e-5, wd=0.01) |
| Scheduler | Linear warmup (10%) + decay |
| Epochs | 3 (quick: 1) |
| Batch size | 32 (quick: 16) |
| Early stopping | patience=3 |
| Gradient clipping | max norm=1.0 |

---

## UIT-VSFC — Manual Download (Optional)

The code automatically tries to fetch UIT-VSFC from GitHub.
If your network blocks raw GitHub, place the files manually:

```
data/vsfc/
├── train/
│   ├── sents.txt        ← one sentence per line
│   └── sentiments.txt   ← one label (0/1/2) per line
├── dev/
│   ├── sents.txt
│   └── sentiments.txt
└── test/
    ├── sents.txt
    └── sentiments.txt
```

Download from: https://github.com/nguyenlab-sfl/UIT-VSFC

---

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
accelerate>=0.24.0
sentencepiece>=0.1.99
tqdm>=4.66.0
requests>=2.31.0
```

---

## GPU Recommendations

| Hardware | Estimated training time (full) |
|----------|-------------------------------|
| NVIDIA RTX 3080 / 4080 | ~30–60 min |
| NVIDIA T4 (Colab free) | ~90–120 min |
| CPU only | ~6–12 h (use `--quick`) |

---

## License

MIT — free to use and modify for educational and research purposes.
