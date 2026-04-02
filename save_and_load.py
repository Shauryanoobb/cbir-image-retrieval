import pickle

def save_feature_db(feature_db, filename):
    with open(filename, "wb") as f:
        pickle.dump(feature_db, f)

def load_feature_db(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)