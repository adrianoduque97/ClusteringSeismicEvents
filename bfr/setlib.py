""" This module contains functions mainly operating on/with
the discard/compress/retain sets of a bfr.Model"""

from . import clustlib
from . import ptlib


def try_retain(point, model):
    """ Updates the retain set of model according to point. If the point is considered near
    the centroid of a cluster in retain, the clusters get merged and moved to the compress set.
    Distance, threshold and threshold function is given by the defaults of model.

    ----------
    point : numpy.ndarray

    model : bfr.Model

    Returns
    -------

    """

    new_cluster = clustlib.Cluster(model.dimensions)
    clustlib.update_cluster(point, new_cluster)
    if not model.retain:
        model.retain.append(new_cluster)
        return
    closest_idx = clustlib.closest(point, model.retain, clustlib.euclidean)
    model.retain[0], model.retain[closest_idx] = model.retain[closest_idx], model.retain[0]

    closest_cluster = model.retain[0]
    if clustlib.euclidean(point, closest_cluster) < model.eucl_threshold:
        model.retain.pop(0)
        clustlib.update_cluster(point, closest_cluster)
        model.compress.append(closest_cluster)
    else:
        model.retain.append(new_cluster)


def try_include(point, cluster_set, model):
    """ Includes a point in the closest cluster of cluster_set if it is considered close.
    Distance, threshold and threshold function is given by the defaults of model.

    Parameters
    ----------
    point : numpy.ndarray

    cluster_set : list
        the list of clusters to try

    model : bfr.model
        Default threshold and threshold_fun settings of the model determine if
        a point will be included with the set.

    Returns
    -------
    bool
        True if the point is assigned to a cluster. False otherwise

    """

    if ptlib.used(point):
        return True
    if not cluster_set:
        return False
    closest_idx = clustlib.closest(point, cluster_set, model.distance_fun)
    closest_cluster = cluster_set[closest_idx]
    if model.threshold_fun(point, closest_cluster) < model.threshold:
        clustlib.update_cluster(point, closest_cluster)
        return True
    return False


def finalize_set(cluster_set, model):
    """ Assigns the clusters in cluster_set to the closest cluster in the discard set of model.

    ----------
    cluster_set : list
        the list of clusters to finalize

    model : bfr.Model

    Returns
    -------

    """

    for cluster in cluster_set:
        mean = clustlib.mean(cluster)
        closest_idx = clustlib.closest(mean, model.discard, model.distance_fun)
        closest_cluster = model.discard[closest_idx]
        merged = clustlib.merge_clusters(cluster, closest_cluster)
        model.discard[closest_idx] = merged


def update_compress(model):
    """ Updates the compress set by merging all clusters in the compress set which have
    a merged std_dev <= (std_dev(compress_cluster) + std_dev(other_compress_cluster)) *
    model.merge_threshold

    Parameters
    ----------
    model : bfr.Model

    Returns
    -------

    """

    if len(model.compress) == 1:
        return
    for each in model.compress:
        cluster = model.compress.pop(0)
        centroid = clustlib.mean(cluster)
        closest_idx = clustlib.closest(centroid, model.compress, clustlib.mahalanobis)
        closest_cluster = model.compress[closest_idx]
        if clustlib.std_check(cluster, closest_cluster, model.merge_threshold):
            merged = clustlib.merge_clusters(cluster, closest_cluster)
            model.compress[closest_idx] = merged
            return update_compress(model)
        model.compress.append(cluster)
