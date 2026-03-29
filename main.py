from feature_extraction import extract_color_histogram, extract_lbp, extract_lbp_fast
from retrieval import build_feature_database, retrieve
from evaluation import precision_at_k
from visualize import show_results
import os

DATASET_PATH = "dataset"
#FEATURE_EXTRACTOR = extract_color_histogram
FEATURE_EXTRACTOR = extract_lbp
#FEATURE_EXTRACTOR = extract_lbp_fast


def main():
    print("Building feature database...")
    feature_db = build_feature_database(DATASET_PATH, FEATURE_EXTRACTOR)

    query_image = "100.jpg"
    query_path = os.path.join(DATASET_PATH, query_image)

    results = retrieve(query_path, feature_db, FEATURE_EXTRACTOR, top_k=10)

    precision = precision_at_k(results, query_image, k=10)
    show_results(query_path, results, DATASET_PATH)
    print(f"\nPrecision@10: {precision:.4f}")


if __name__ == "__main__":
    main()