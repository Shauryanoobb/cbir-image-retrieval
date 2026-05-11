from feature_extraction import extract_color_histogram, extract_lbp, extract_lbp_fast, extract_glcm_features
from combined_feature_extraction import (
    extract_color_histogram_glcm_features,
    extract_color_histogram_lbp_fast,
    extract_color_histogram_lbp_fast_glcm_features,
    extract_lbp_fast_glcm_features,
)
from similarity import euclidean_distance, cosine_distance
from retrieval import build_feature_database, retrieve
from evaluation import precision_at_k
from visualize import show_results
from save_and_load import save_feature_db, load_feature_db
import os

DATASET_PATH = "dataset"

# ── Feature extractor toggle ──────────────────────────────────────────────────
# FEATURE_EXTRACTOR = extract_color_histogram
# FEATURE_EXTRACTOR = extract_lbp
# FEATURE_EXTRACTOR = extract_lbp_fast
# FEATURE_EXTRACTOR = extract_glcm_features
# FEATURE_EXTRACTOR = extract_color_histogram_lbp_fast
# FEATURE_EXTRACTOR = extract_color_histogram_glcm_features
# FEATURE_EXTRACTOR = extract_lbp_fast_glcm_features
FEATURE_EXTRACTOR = extract_color_histogram_lbp_fast_glcm_features

# ── Distance metric toggle ────────────────────────────────────────────────────
DISTANCE_METRIC = euclidean_distance
# DISTANCE_METRIC = cosine_distance

FEATURE_FILE = f"feature_files/features_{FEATURE_EXTRACTOR.__name__}.pkl"


def main():
    if not os.path.exists(FEATURE_FILE):
        print("Building feature database (one-time)...")
        feature_db = build_feature_database(DATASET_PATH, FEATURE_EXTRACTOR)
        save_feature_db(feature_db, FEATURE_FILE)
    else:
        print("Loading precomputed features...")
        feature_db = load_feature_db(FEATURE_FILE)

    query_image = "990.jpg"
    query_path = os.path.join(DATASET_PATH, query_image)

    results = retrieve(query_path, feature_db, FEATURE_EXTRACTOR, top_k=10, distance_fn=DISTANCE_METRIC)

    precision = precision_at_k(results, query_image, k=10)
    show_results(query_path, results, DATASET_PATH)
    print(f"\nPrecision@10: {precision:.4f}")


if __name__ == "__main__":
    main()