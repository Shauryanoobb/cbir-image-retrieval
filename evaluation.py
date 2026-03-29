def get_class(filename):
    return filename[0]  # first digit


def precision_at_k(results, query_image, k):
    query_class = get_class(query_image)

    relevant = 0

    for filename, _ in results[:k]:
        if get_class(filename) == query_class:
            relevant += 1

    return relevant / k