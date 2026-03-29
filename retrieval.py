import os
from feature_extraction import extract_color_histogram
from similarity import euclidean_distance
from tqdm import tqdm

def build_feature_database(dataset_path):
    features = {}
    
    for filename in tqdm(os.listdir(dataset_path)):
        if filename.endswith(".jpg"):
            path = os.path.join(dataset_path, filename)
            features[filename] = extract_color_histogram(path)
    
    return features


def retrieve(query_path, feature_db, dataset_path, top_k=10):
    query_feature = extract_color_histogram(query_path)

    distances = []

    for filename, feature in feature_db.items():
        dist = euclidean_distance(query_feature, feature)
        distances.append((filename, dist))

    distances.sort(key=lambda x: x[1])

    return distances[:top_k]