"""This module contains objective functions."""


def finish_points(idx=None, points=None, model=None):
    """ Used to determine when all points have been clustered.

    Parameters
    ----------
    idx : int

    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    Returns
    -------
    bool
        True if idx is the last row index of points. False otherwise.

    """

    return idx == len(points) - 1


def zerofree_variances(idx=None, points=None, model=None):
    """ Used to determine when all clusters of the discard set have non zero variance
    in all dimensions

    Parameters
    ----------
    model : bfr.model

    Returns
    -------
    bool
        True if all clusters within the discard set of model has a non zero variance.
        False otherwise

    """

    has_variances = filter(lambda cluster: cluster.has_variance, model.discard)
    with_variances = list(has_variances)
    zerofree_discard = len(with_variances) == len(model.discard)
    return zerofree_discard
