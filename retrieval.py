import os
from similarity import euclidean_distance
from tqdm import tqdm


def build_feature_database(dataset_path, feature_extractor):
    features = {}

    for filename in tqdm(os.listdir(dataset_path)):
        if filename.endswith(".jpg"):
            path = os.path.join(dataset_path, filename)
            features[filename] = feature_extractor(path)

    return features


def retrieve(query_path, feature_db, feature_extractor, top_k=10, distance_fn=euclidean_distance):
    query_feature = feature_extractor(query_path)

    distances = []

    for filename, feature in feature_db.items():
        dist = distance_fn(query_feature, feature)
        distances.append((filename, dist))

    distances.sort(key=lambda x: x[1])

    return distances[:top_k]