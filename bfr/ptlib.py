""" This module contains functions mainly operating on/with points (numpy.ndarrays)"""

import random
import numpy


def sum_squared_diff(point, other_point):
    """ Computes the sum of dimensions of (point - other_point) ^ 2

    Parameters
    ----------
    point : numpy.ndarray

    other_point : numpy.ndarray

    Returns
    -------
    float
        The sum of dimensions of (point - other_point) ^ 2

    """

    diff = point - other_point
    return numpy.dot(diff, diff)


def euclidean(point, other_point):
    """ Computes the euclidean distance between a point and another point
    d(v, w) = ||v - w|| = sqrt(sum(v_i - w_i)^2)

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    other_point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    Returns
    -------
    float
        The Euclidean distance between point and other_point

    """

    sq_diff = sum_squared_diff(point, other_point)
    return numpy.sqrt(sq_diff)


def sum_all_euclideans(points):
    """ Not currently used.

    Parameters
    ----------
    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    Returns
    -------
    float
        The sum of all

    """

    total = 0
    for point in points:
        diffs = points - point
        squared = diffs ** 2
        summed = numpy.sum(squared)
        total += numpy.sqrt(summed)
    return total


def used(point):
    """ Checks if a point has been used as initial point.

    Parameters
    ----------
    point : numpy.ndarray

    Returns
    -------
    bool
        True if the point has been used. Represented by row being numpy.nan

    """

    return numpy.isnan(point[0])


def random_points(points, model, seed=None):
    """ Returns a number of random points from points. Marks the selected points
    as used by setting them to numpy.nan. Not currently used.

    Parameters
    ----------
    nof_points : int
        The number of points to be returned

    model : bfr.Model

    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.
        Used points will be set to numpy.nan

    rounds : int
        The number of initialization rounds

    seed : int
        Sets the random seed

    Returns
    -------
    initial points : numpy.matrix
        The round of random points which maximizes the distance between all

    """

    max_index = len(points) - 1
    random.seed(seed)
    samples = []
    scores = []
    for round_idx in range(model.init_rounds):
        idx = random.sample(range(max_index), model.nof_clusters)
        sample_points = points[idx]
        spread_score = sum_all_euclideans(sample_points)
        samples.append(idx)
        scores.append(spread_score)
    max_dist = numpy.argmax(scores)
    idx = samples[max_dist]
    initial_points = points[idx]
    points[idx] = numpy.nan
    return initial_points


def best_spread(points, model, seed=None):
    """ Optimizes the spread of the initial points by maximizing the minimum distance between
    each point.

    Parameters
    ----------
    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.
        Used points will be set to numpy.nan

    model : bfr.Model

    seed : int
        Sets the random seed

    Returns
    -------
    initial_points : list
        Each point in the list maximizes the distance to the closest of the other points

    """

    max_index = len(points) - 1
    random.seed(seed)
    so_far = random.sample(range(max_index), 1)
    for cluster_idx in range(model.nof_clusters - 1):
        candidate_idx = random.sample(range(max_index), model.init_rounds)
        idx = max_mindist(points, so_far, candidate_idx)
        so_far.append(idx)
    initial_points = points[so_far]
    points[so_far] = numpy.nan
    return initial_points


def max_mindist(points, so_far, candidates):
    """ Finds the candidate index which maximizes the distance to the points
    with indices in so_far

    Parameters
    ----------
    points : numpy.ndarray
        The points from which the random points will be picked.
        Randomly picked points will be set to numpy.nan

    so_far : list of ints
        The points already chosen

    candidates : list of ints
        Randomly picked indices of candidates

    Returns
    -------
    index : int
        The index of the candidate with the highest minimum distance to
        the points which indices are in so_far

    """

    distances = numpy.zeros((len(candidates), len(so_far)))
    for row, candidate_idx in enumerate(candidates):
        for column, chosen_idx in enumerate(so_far):
            chose = points[chosen_idx]
            candidate = points[candidate_idx]
            distances[row][column] = euclidean(chose, candidate)
    mins = numpy.amin(distances, axis=1)
    highest_min_idx = numpy.argmax(mins)
    return candidates[highest_min_idx]
