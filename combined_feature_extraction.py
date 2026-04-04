import numpy as np

from feature_extraction import (
    extract_color_histogram,
    extract_glcm_features,
    extract_lbp_fast,
)


def _normalize_feature_vector(feature_vector):
    feature_vector = np.asarray(feature_vector, dtype=np.float32)
    norm = np.linalg.norm(feature_vector)

    if norm > 0:
        return feature_vector / norm

    return feature_vector


def _combine_features(image_path, extractors):
    feature_vectors = [
        _normalize_feature_vector(extractor(image_path))
        for extractor in extractors
    ]
    return np.concatenate(feature_vectors)


def extract_color_histogram_lbp_fast(image_path):
    return _combine_features(
        image_path,
        [extract_color_histogram, extract_lbp_fast],
    )


def extract_color_histogram_glcm_features(image_path):
    return _combine_features(
        image_path,
        [extract_color_histogram, extract_glcm_features],
    )


def extract_lbp_fast_glcm_features(image_path):
    return _combine_features(
        image_path,
        [extract_lbp_fast, extract_glcm_features],
    )


def extract_color_histogram_lbp_fast_glcm_features(image_path):
    return _combine_features(
        image_path,
        [extract_color_histogram, extract_lbp_fast, extract_glcm_features],
    )
