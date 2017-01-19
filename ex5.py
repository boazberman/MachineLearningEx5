import sys
from PIL import Image
import numpy as np

K_MEANS_NUMBER_ITERATIONS = 10

def parse_data(initial_centroids_path, image_file_path):
    im = Image.open(image_file_path)
    x = np.array(im)
    centroids = np.loadtxt(initial_centroids_path)

    return centroids,x

def calc_new_centroids(centroids, image):
    clusters_pixels = {}
    for i in xrange(K_MEANS_NUMBER_ITERATIONS):
        # classify to clusters
        for row in image:
            for pixel in row:
                cluster_id = np.argmin([np.linalg.norm(pixel - c) for c in centroids])
                if cluster_id not in clusters_pixels:
                    clusters_pixels[cluster_id] = []
                clusters_pixels[cluster_id].append(pixel)
        # update centroids
        for idx,c in enumerate(centroids):
            if len(clusters_pixels[idx]) != 0:
                centroids[idx] = np.sum([arr for arr in clusters_pixels[idx]], axis=0) / len(clusters_pixels[idx])
        clusters_pixels = {}

    for c in centroids:
        print "%s" % c

    return centroids

def find_nearest_centroid(pixel, centroids):
    return centroids[np.argmin([np.linalg.norm(pixel - c) for c in centroids])]


def calc_new_image(image, centroids):
    new_image = image
    for image_idx,row in enumerate(image):
        new_row = row
        for row_idx,pixel in enumerate(row):
            new_row[row_idx] = find_nearest_centroid(pixel,centroids)
        new_image[image_idx] = new_row

    return new_image


def save_new_image(new_image,image_file_name):
    x = Image.fromarray(new_image)
    x.save(image_file_name+'_comp.tif')


def extract_file_name(file_path):
    if "\\" in file_path:
        return file_path.rsplit("\\",1)[1].split(".")[0]
    return file_path.split(".")[0]

if __name__ == "__main__":
    initial_centroids_path = sys.argv[1]
    image_file_path = sys.argv[2]
    centroids,image = parse_data(initial_centroids_path, image_file_path)
    new_centroids = calc_new_centroids(centroids, image)
    new_image = calc_new_image(image, new_centroids)
    save_new_image(new_image, extract_file_name(image_file_path))
