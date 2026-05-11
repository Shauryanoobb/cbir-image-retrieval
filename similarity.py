import numpy as np


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def cosine_distance(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1.0 - np.dot(vec1, vec2) / (norm1 * norm2)