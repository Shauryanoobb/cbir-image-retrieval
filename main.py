import os
from retrieval import build_feature_database, retrieve
from evaluation import precision_at_k
from visualize import show_results

DATASET_PATH = "dataset"   # your folder

def main():
    print("Building feature database...")
    feature_db = build_feature_database(DATASET_PATH)

    # pick any query image
    query_image = "100.jpg"
    query_path = os.path.join(DATASET_PATH, query_image)

    print(f"\nQuery Image: {query_image}")

    results = retrieve(query_path, feature_db, DATASET_PATH, top_k=10)

    print("\nTop 10 Results:")
    for filename, dist in results:
        print(f"{filename} - Distance: {dist:.4f}")

    precision = precision_at_k(results, query_image, k=10)
    print(f"\nPrecision@10: {precision:.4f}")

    show_results(query_path, results, DATASET_PATH)


if __name__ == "__main__":
    main()