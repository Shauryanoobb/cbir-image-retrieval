from feature_extraction import extract_color_histogram, extract_lbp, extract_lbp_fast, extract_glcm_features
from retrieval import build_feature_database, retrieve
from evaluation import precision_at_k
from visualize import show_results
from save_and_load import save_feature_db, load_feature_db
import os

DATASET_PATH = "dataset"

# FEATURE_EXTRACTOR = extract_color_histogram
# FEATURE_EXTRACTOR = extract_lbp
# FEATURE_EXTRACTOR = extract_lbp_fast
FEATURE_EXTRACTOR = extract_glcm_features

FEATURE_FILE = f"feature_files/features_{FEATURE_EXTRACTOR.__name__}.pkl"

#TAKE CARE WITH GITIGNORE NOW, small pkl file not a problem

def main():
    # print("Building feature database...")
    # feature_db = build_feature_database(DATASET_PATH, FEATURE_EXTRACTOR)
    # 🔁 Build only if not exists
    if not os.path.exists(FEATURE_FILE):
        print("Building feature database (one-time)...")
        feature_db = build_feature_database(DATASET_PATH, FEATURE_EXTRACTOR)
        save_feature_db(feature_db, FEATURE_FILE)
    else:
        print("Loading precomputed features...")
        feature_db = load_feature_db(FEATURE_FILE)

    query_image = "10.jpg"
    query_path = os.path.join(DATASET_PATH, query_image)

    results = retrieve(query_path, feature_db, FEATURE_EXTRACTOR, top_k=10)

    precision = precision_at_k(results, query_image, k=10)
    show_results(query_path, results, DATASET_PATH)
    print(f"\nPrecision@10: {precision:.4f}")


if __name__ == "__main__":
    main()