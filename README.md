# Content-Based Image Retrieval (CBIR)

A minimal implementation of a Content-Based Image Retrieval (CBIR) system using:

* Multiple feature extraction methods
* Euclidean Distance (similarity)
* Precision@K (evaluation)

This project is designed to stay simple while supporting multiple feature extraction strategies, including combined feature representations built by concatenating existing feature vectors.

---

## Current Status

* Features: Color Histogram, LBP, Fast LBP, and GLCM
* Combined features: every combination of the main implemented extractors used in the current pipeline
* Single distance metric: Euclidean
* Single evaluation metric: Precision@K
* Feature databases are cached in `feature_files/` and loaded on later runs

---

## Project Structure

``` 
cbir/
│
├── dataset/                  # Wang dataset (1000 images, not included in repo)
│
├── feature_extraction.py     # Individual feature extractors
├── combined_feature_extraction.py  # Combined feature extractors
├── save_and_load.py          # Save/load precomputed feature databases
├── similarity.py             # Euclidean distance
├── retrieval.py              # Retrieval pipeline (build + search)
├── evaluation.py             # Precision@K computation
├── visualize.py              # Visualization of results
├── main.py                   # Entry point for selecting extractor + running retrieval
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

1. Select a feature extractor in `main.py`
2. Represent images as feature vectors
3. Build the feature database if it does not already exist for that extractor
4. Compare the query image with the database using Euclidean distance
5. Rank images based on similarity
6. Return the top-K results
7. Evaluate using Precision@K

---

## Available Feature Extractors

The project currently supports these individual extractors:

* `extract_color_histogram`
* `extract_lbp`
* `extract_lbp_fast`
* `extract_glcm_features`

The project also supports these combined extractors from `combined_feature_extraction.py`:

* `extract_color_histogram_lbp_fast`
* `extract_color_histogram_glcm_features`
* `extract_lbp_fast_glcm_features`
* `extract_color_histogram_lbp_fast_glcm_features`

The combined extractors work by concatenating the output vectors from the selected individual extractors into one larger feature vector.

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

To switch the active feature extractor, edit the `FEATURE_EXTRACTOR` assignment in `main.py`.

Example:

```python
# FEATURE_EXTRACTOR = extract_glcm_features
FEATURE_EXTRACTOR = extract_color_histogram_lbp_fast_glcm_features
```

The feature file is stored automatically as:

```python
feature_files/features_{FEATURE_EXTRACTOR.__name__}.pkl
```

This means each extractor, including combined ones, gets its own cached feature database.

---

## Notes

* The dataset is not included in the repository and must be placed manually in the `dataset/` folder
* `extract_lbp` is the slower reference-style LBP implementation, while `extract_lbp_fast` is the practical optimized version for regular use
* Combined feature extraction can improve retrieval quality by mixing color and texture information, but it also increases feature dimensionality and storage size
* Retrieval quality still depends on the selected feature representation and the use of Euclidean distance as the similarity metric
