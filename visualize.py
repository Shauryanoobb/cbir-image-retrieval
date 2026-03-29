import cv2
import matplotlib.pyplot as plt
import os

def show_results(query_path, results, dataset_path):
    plt.figure(figsize=(12, 4))

    # Show query
    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis("off")

    # Show results
    for i, (filename, _) in enumerate(results):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        plt.subplot(1, len(results) + 1, i + 2)
        plt.imshow(img)
        plt.title(filename)
        plt.axis("off")

    plt.show()