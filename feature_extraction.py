import cv2
import numpy as np

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

    # Normalize
    hist = cv2.normalize(hist, hist).flatten()

    return hist