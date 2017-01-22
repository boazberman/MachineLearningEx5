import sys
from PIL import Image
import numpy as np

ITERATIONS = 10


def classify(centroids, image):
    clusters = {}
    for row in image:
        for pixel in row:
            cluster_id = np.argmin([np.linalg.norm(pixel - c) for c in centroids])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pixel)
    return clusters


def recalc_image(image, centroids):
    for row in image:
        for i, pixel in enumerate(row):
            row[i] = find_nearest_centroid(pixel, centroids)
    return image


def recalculate_centroids(centroids, clusters_pixels):
    for i in xrange(len(centroids)):
        if len(clusters_pixels[i]) != 0:
            centroids[i] = np.sum(clusters_pixels[i], axis=0) / float(len(clusters_pixels[i]))


def find_nearest_centroid(pixel, centroids):
    return centroids[np.argmin([np.linalg.norm(pixel - c) for c in centroids])]


def save_new_image(new_image, image_file_name):
    Image.fromarray(new_image).save(image_file_name + '_comp.tif')


def calculate_centroids(centroids, image):
    for _ in xrange(ITERATIONS):
        clustered_pixels = classify(centroids, image)
        recalculate_centroids(centroids, clustered_pixels)

    for row in centroids:
        print "%s" % " ".join("%s" % int(cell) for cell in row)

    return centroids


def to_output_file_name(file_path):
    if "\\" in file_path:
        file_path = file_path.rsplit("\\", 1)[1]
    return file_path.rsplit(".", 1)[0]


def open_and_parse(initial_centroids_path, image_file_path):
    centroids = np.loadtxt(initial_centroids_path)
    x = np.array(Image.open(image_file_path))

    return centroids, x


def main(args):
    centroids_file = args[0]
    image_file = args[1]
    initial_centroids, image = open_and_parse(centroids_file, image_file)
    final_centroids = calculate_centroids(initial_centroids, image)
    save_new_image(recalc_image(image, final_centroids), to_output_file_name(image_file))


if __name__ == "__main__":
    main(sys.argv[1:])
