"""
Comprehensive Precision@K evaluation for all CBIR feature extractors.

Runs every image as a query (leave-one-out), excludes the query image from
retrieved results, then reports global and per-class Precision@K.

Usage:
    python evaluate_all.py
    python evaluate_all.py --skip-slow   # skip extract_lbp (very slow)
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import (
    extract_color_histogram,
    extract_lbp,
    extract_lbp_fast,
    extract_glcm_features,
)
from combined_feature_extraction import (
    extract_color_histogram_lbp_fast,
    extract_color_histogram_glcm_features,
    extract_lbp_fast_glcm_features,
    extract_color_histogram_lbp_fast_glcm_features,
)
from retrieval import build_feature_database, retrieve
from save_and_load import save_feature_db, load_feature_db
from evaluation import get_class

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = "dataset"
FEATURE_FILES_DIR = "feature_files"
RESULTS_CSV = "evaluation_results.csv"

K_VALUES = [1, 5, 10, 20]

CLASS_NAMES = {
    0: "Africans",
    1: "Beaches",
    2: "Buildings",
    3: "Buses",
    4: "Dinosaurs",
    5: "Elephants",
    6: "Flowers",
    7: "Horses",
    8: "Mountains",
    9: "Food",
}

FAST_EXTRACTORS = [
    extract_color_histogram,
    extract_lbp_fast,
    extract_glcm_features,
    extract_color_histogram_lbp_fast,
    extract_color_histogram_glcm_features,
    extract_lbp_fast_glcm_features,
    extract_color_histogram_lbp_fast_glcm_features,
]

SLOW_EXTRACTORS = [extract_lbp]

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_or_build_feature_db(extractor):
    feature_file = os.path.join(
        FEATURE_FILES_DIR, f"features_{extractor.__name__}.pkl"
    )
    if os.path.exists(feature_file):
        return load_feature_db(feature_file)
    print(f"  Building feature database (one-time)...")
    db = build_feature_database(DATASET_PATH, extractor)
    save_feature_db(db, feature_file)
    return db


def precision_at_k_excl_query(results, query_image, k):
    """Precision@K with the query image excluded from retrieved results."""
    query_class = get_class(query_image)
    filtered = [f for f, _ in results if f != query_image]
    relevant = sum(1 for f in filtered[:k] if get_class(f) == query_class)
    return relevant / k


def evaluate_extractor(extractor, feature_db, all_images):
    """
    Query every image, compute Precision@K for each K in K_VALUES.

    Returns
    -------
    global_precision : dict  {k: float}
    class_precision  : dict  {class_id: {k: float}}
    """
    max_k = max(K_VALUES)
    # class_scores[cls][k] = [score, score, ...]
    class_scores = defaultdict(lambda: defaultdict(list))

    for query_image in tqdm(all_images, desc="  queries", ncols=72, leave=False):
        query_path = os.path.join(DATASET_PATH, query_image)
        query_class = get_class(query_image)
        # +1 so we can drop the self-match and still have max_k results
        results = retrieve(query_path, feature_db, extractor, top_k=max_k + 1)
        for k in K_VALUES:
            p = precision_at_k_excl_query(results, query_image, k)
            class_scores[query_class][k].append(p)

    global_precision = {}
    class_precision = {}

    for k in K_VALUES:
        all_scores = []
        for cls in sorted(class_scores):
            scores = class_scores[cls][k]
            class_precision.setdefault(cls, {})[k] = float(np.mean(scores))
            all_scores.extend(scores)
        global_precision[k] = float(np.mean(all_scores))

    return global_precision, class_precision


# ── Printing ──────────────────────────────────────────────────────────────────

COL = 9  # width per P@K column


def _header_row(k_values):
    return "".join(f"{'P@'+str(k):>{COL}}" for k in sorted(k_values))


def print_global_table(all_results):
    k_values = sorted(K_VALUES)
    name_w = max(len(n) for n in all_results) + 2

    sep = "=" * (name_w + COL * len(k_values))
    print(sep)
    print("GLOBAL PRECISION@K  (averaged over all 1000 queries)")
    print(sep)
    print(f"{'Extractor':<{name_w}}" + _header_row(k_values))
    print("-" * len(sep))

    for name, data in all_results.items():
        row = f"{name:<{name_w}}"
        row += "".join(f"{data['global'][k]:>{COL}.4f}" for k in k_values)
        print(row)

    print(sep)


def print_class_table(extractor_name, class_precision):
    k_values = sorted(K_VALUES)
    name_w = max(len(v) for v in CLASS_NAMES.values()) + 2

    print(f"\n  [{extractor_name}]")
    inner_sep = "-" * (name_w + COL * len(k_values))
    print("  " + inner_sep)
    print("  " + f"{'Class':<{name_w}}" + _header_row(k_values))
    print("  " + inner_sep)

    for cls_id in sorted(class_precision):
        name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
        row = f"{name:<{name_w}}"
        row += "".join(
            f"{class_precision[cls_id][k]:>{COL}.4f}" for k in k_values
        )
        print("  " + row)

    print("  " + inner_sep)


# ── CSV export ────────────────────────────────────────────────────────────────


def save_csv(all_results, path):
    k_values = sorted(K_VALUES)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(["GLOBAL RESULTS"])
        w.writerow(["Extractor"] + [f"P@{k}" for k in k_values])
        for name, data in all_results.items():
            w.writerow([name] + [f"{data['global'][k]:.4f}" for k in k_values])

        w.writerow([])
        w.writerow(["PER-CLASS RESULTS"])
        for name, data in all_results.items():
            w.writerow([f"--- {name} ---"])
            w.writerow(["Class"] + [f"P@{k}" for k in k_values])
            for cls_id in sorted(data["class"]):
                cls_name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
                w.writerow(
                    [cls_name]
                    + [f"{data['class'][cls_id][k]:.4f}" for k in k_values]
                )
            w.writerow([])


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate all CBIR feature extractors")
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip extract_lbp (pixel-loop implementation, very slow on 1000 images)",
    )
    args = parser.parse_args()

    os.makedirs(FEATURE_FILES_DIR, exist_ok=True)

    extractors = FAST_EXTRACTORS + ([] if args.skip_slow else SLOW_EXTRACTORS)

    all_images = sorted(
        [f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )

    n_classes = len(set(get_class(f) for f in all_images))
    print(f"Dataset  : {len(all_images)} images, {n_classes} classes")
    print(f"K values : {K_VALUES}")
    print(f"Extractors: {len(extractors)}")
    print()

    all_results = {}

    for extractor in extractors:
        name = extractor.__name__
        print(f"[{name}]")
        t0 = time.time()

        feature_db = get_or_build_feature_db(extractor)
        global_p, class_p = evaluate_extractor(extractor, feature_db, all_images)

        elapsed = time.time() - t0
        summary = "  " + "  ".join(
            f"P@{k}={global_p[k]:.4f}" for k in sorted(K_VALUES)
        )
        print(f"{summary}   ({elapsed:.1f}s)")

        all_results[name] = {"global": global_p, "class": class_p}

    print()
    print_global_table(all_results)

    print()
    print("=" * 60)
    print("PER-CLASS PRECISION@K")
    print("=" * 60)
    for name, data in all_results.items():
        print_class_table(name, data["class"])

    save_csv(all_results, RESULTS_CSV)
    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
