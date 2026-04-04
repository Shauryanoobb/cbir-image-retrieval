def get_class(filename):
    image_number = int(filename.split(".")[0])
    return image_number // 100


def precision_at_k(results, query_image, k):
    query_class = get_class(query_image)

    relevant = 0

    for filename, _ in results[:k]:
        if get_class(filename) == query_class:
            relevant += 1

    return relevant / k