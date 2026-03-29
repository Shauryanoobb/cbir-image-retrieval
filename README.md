# Content-Based Image Retrieval (CBIR)

A minimal implementation of a Content-Based Image Retrieval (CBIR) system using:

* Color Histogram (feature)
* Euclidean Distance (similarity)
* Precision@K (evaluation)

This is a baseline version designed to be simple and extensible. The goal is to later generalize the pipeline to support multiple features, distance metrics, and evaluation strategies.

---

## Current Status

* Single feature: Color Histogram
* Single distance metric: Euclidean
* Single evaluation metric: Precision@K
* The feature database is rebuilt on every run (to be optimized later)

---

## Project Structure

```
cbir/
│
├── dataset/                  # Wang dataset (1000 images, not included in repo)
│
├── feature_extraction.py     # Color histogram extraction
├── similarity.py             # Euclidean distance
├── retrieval.py              # Retrieval pipeline (build + search)
├── evaluation.py             # Precision@K computation
├── visualise.py              # Visualization of results
├── main.py                   # Entry point (will be configurable later)
│
└── requirements.txt
```

---

## Wang Dataset Overview

* Total images: 1000
* Categories: 10
* Images per category: 100

### Label Encoding

The dataset does not provide explicit labels, but they are encoded in filenames:

```
0xx.jpg → Class 0 (Africans)
1xx.jpg → Class 1 (Beaches)
2xx.jpg → Class 2 (Buildings)
...
9xx.jpg → Class 9 (Food)
```

Example:

* `100.jpg` → Class 1 (Beaches)
* `257.jpg` → Class 2 (Buildings)

This encoding is used for evaluation with Precision@K.

---

## How It Works

1. Extract color histogram features for all images
2. Represent images as feature vectors
3. Compare the query image with the database using Euclidean distance
4. Rank images based on similarity
5. Return the top-K results
6. Evaluate using Precision@K

---

## Visualization

The system includes a visualization utility to inspect retrieval results:

* Displays the query image alongside the top-K retrieved images
* Helps analyze the strengths and limitations of the current approach

---

## How to Run

```bash
python main.py
```

---

## Notes

* The dataset is not included in the repository and must be placed manually in the `dataset/` folder
* The current implementation relies only on color features and does not capture semantic similarity effectively
* Performance is expected to improve with the addition of texture features and combined representations
