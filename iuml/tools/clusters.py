from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import cv2
from numba import jit

def get_clustering_points(img):
    '''
    Retrieve points that need to be clustered from a grayscale image
    Parameters:
        img - grayscale img
    Returns:
        coordinates of each non-zero pixel in the mask (row, col)
    '''
    rows, cols = np.nonzero(img)
    return list(zip(rows, cols))

def get_number_of_clusters(labels):
    '''
    From a set of an array where each element is a cluster label
    Extract the number of clusters
    '''
    return len(set(labels)) - (1 if -1 in labels else 0)

def cluster_dbscan(img, eps = 3, min_samples=10):
    '''
    Apply clustering.
    Parameters:
        img -- binary mask where
        eps -- distance needed to be included in the cluster
        min_samples - how many samples make a cluster
    Returns:
        X, labels, n_clusters
        X - coordinates of each non-zero pixel in the mask (row, col)
        labels - cluster labels
        n_clusters - total clusters
    '''
    labels = []
    n_clusters = 0

    X = get_clustering_points(img)
    if len(X) == 0:
        return X, labels, n_clusters

    algo = DBSCAN(eps = eps, min_samples = min_samples)
    try:
        labels = algo.fit_predict(X)
        n_clusters = get_number_of_clusters(labels)
    except:
        pass

    return X, labels, n_clusters

def cluster_kmeans(img, n_classes=2):
    '''
    Apply k-means clustering.
    Parameters:
        img -- binary mask where
        n_clusters - number of classes
    Returns:
        X, labels, n_clusters
        X - coordinates of each non-zero pixel in the mask (row, col)
        labels - cluster labels
    '''
    X = get_clustering_points(img)
    kmeans = KMeans(n_clusters = n_classes)
    labels = kmeans.fit_predict(X)
    return X, labels

@jit
def min_max_coords(rect1, pt2):
    (l1, t1, r1, b1) = rect1
    (l2, t2) = pt2

    l = min(l1, l2)
    t = min(t1, t2)
    r = max(r1, l2)
    b = max(b1, t2)
    return l, t, r, b

@jit
def map_clusters_to_bounding_boxes(labels, coords, n_clusters = None):
    '''
    Map each cluster to its bounding box
    '''
    if n_clusters is None:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    buds = [None] * n_clusters
    for label, coord in zip(labels, coords):
        if label == -1:
            continue

        t, l = coord

        if buds[label] is None:
            buds[label] = (l, t, l, t)
        else:
            buds[label] = min_max_coords(buds[label], (l, t))

    return buds


def mask_from_clusters(labels, coords, mask_shape_row_col):
    '''
    Create a 2D mask where each (row, col) is set to the label of its cluster
    or 0 if it is non-clustered

    Parameters:
        labels - array-like where labels[i] == cluster_number
        coords - coordinates of each element in labels[i]
        mask_shape_row_col - tuple of (rows, cols) for the output mask

    Returns:
        A 2D image where each pixel is set to the number of the cluster + 1
        and un-clustered pixels are set to 0
    '''

    mask = np.zeros(mask_shape_row_col)
    mask[tuple(zip(*coords))] = labels + 1
    return mask

def get_box_center (rect):
    l, t, r, b = rect
    return ((l + r) // 2, (t + b) // 2)

def get_box_radius(rect):
    l, t, r, b = rect
    return max((r-l) // 2, (b - t) // 2)

def draw_clusters(img, clusters, color=(255, 255, 0), radius = None):
    for bud in clusters:
        if radius is None:
            rad = get_box_radius(bud)
        else:
            rad = radius
        _ = cv2.circle(img, get_box_center(bud), rad, color, 2)

def get_clusters_from_mask_verbose(mask, eps = 1, min_samples = 1):
    '''
    Apply DBSCAN clustering to a mask
    Parameters:
        mask - the mask to compute clusters from
        eps - max radius around a given point so points within the circle of this radius are included in the same cluster
        min_samples - minimal number of points to define a cluster
    Returns:
        centers, radii, non-zero pixel coordinates, array of labels, number of clusters, average radius
    '''
    coords, clusters, n_clusters = cluster_dbscan(mask, eps=eps, min_samples = min_samples)
    if n_clusters == 0:
        return None, None, None, None, 0, None

    clustered_rects = map_clusters_to_bounding_boxes(clusters, coords, n_clusters)

    radii = np.array(list(map(get_box_radius, clustered_rects)))
    centers = np.array(list(map(get_box_center, clustered_rects)))
    avg_radius = np.mean(radii)

    return centers, radii, coords, clusters, n_clusters, avg_radius

def get_clusters_from_mask(mask, eps = 1, min_samples = 1):
    '''
    Apply DBSCAN clustering to a mask
    Parameters:
        mask - the mask to compute clusters from
        eps - max radius around a given point so points within the circle of this radius are included in the same cluster
        min_samples - minimal number of points to define a cluster
    Returns:
        centers, radii, non-zero pixel coordinates, array of labels, number of clusters, average radius
    '''

    centers, radii, coords, clusters, n_clusters, avg_radius = get_clusters_from_mask_verbose(mask, eps, min_samples)
    return centers, radii