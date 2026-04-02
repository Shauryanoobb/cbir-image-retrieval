import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        bins,
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist

# if database is ready, then on query we can run this fast, i think conceptually this gets you better understanding of how LBP works, compare dimensionally as well
# ------------------ SLOW LBP (for understanding) ------------------


#study about lbp, so you can decide the best implementation
def extract_lbp(image_path, num_points=24, radius=3):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = np.zeros_like(gray, dtype=np.uint8)

    for i in range(radius, gray.shape[0] - radius):
        for j in range(radius, gray.shape[1] - radius):
            center = gray[i, j]
            code = 0

            for n in range(num_points):
                theta = 2 * np.pi * n / num_points
                x = int(i + radius * np.sin(theta))
                y = int(j + radius * np.cos(theta))

                code = (code << 1) | (gray[x, y] > center)

            lbp[i, j] = code

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist


# ------------------ FAST LBP (use this) ------------------

def extract_lbp_fast(image_path, num_points=24, radius=3):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")

    # number of bins for uniform LBP
    n_bins = num_points + 2

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist



# def extract_glcm_features(image_path, distances=[1], angles=[0], levels=256):
def extract_glcm_features(image_path, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure values are within levels
    gray = (gray / (256 / levels)).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    features = []

    # Extract standard properties
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    for prop in props:
        val = graycoprops(glcm, prop)
        features.extend(val.flatten())

    return np.array(features)