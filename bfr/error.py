""" This module contains the attribute checking of a bfr.Model. If a confirm_*
function evaluates to true, the model should be able to perform the called method.
A confirm method should be called in all methods of a bfr.Model"""

import sys
import numpy


def confirm_attributes(model):
    check_attributes(model)


def confirm_initial_fit(points, model):
    check_initial_fit(points, model)


def confirm_initialized_fit(points, model):
    check_initialized_fit(points, model)


def confirm_predict(points, model):
    check_clusters(points, model)


def confirm_error(points, model):
    if points is None:
        check_attributes(model)
    else:
        check_clusters(points, model)


def confirm_centers(model):
    check_attributes(model)


def confirm_plot(points, model):
    assert model.dimensions == 2 or model.dimensions == 3, "Can only plot 2d or 3d"
    assert model.initialized, "Can only plot initialized models"
    if points is not None:
        check_clusters(points, model)
    else:
        check_attributes(model)


def check_attributes(model):
    assert isinstance(model.mahal_threshold, float), "mahalanobis_factor not float"
    assert isinstance(model.eucl_threshold, float), "euclidean_threshold not float"
    assert isinstance(model.merge_threshold, float), "merge_threshold not float"
    assert isinstance(model.init_rounds, int), "init_rounds not int"
    assert isinstance(model.dimensions, int), "dimensions not int"
    assert isinstance(model.nof_clusters, int), "nof_clusters not int"
    assert model.mahal_threshold > 0, "mahalanobis threshold not > 0"
    assert model.eucl_threshold > 0, "euclidean_threshold not > 0"
    assert model.merge_threshold > 0, "merge_threshold not > 0"
    assert model.dimensions > 0, "dimensions not > 0"
    assert model.nof_clusters > 1, "nof_clusters not > 1"
    assert model.init_rounds > 0, "init_rounds not > 0"


def check_initial_fit(points, model):
    required_nr = model.nof_clusters * model.init_rounds
    check_points(points, model, required_nr)
    check_attributes(model)


def check_initialized_fit(points, model):
    check_points(points, model, 1)
    check_attributes(model)


def check_clusters(points, model):
    check_points(points, model, 1)
    check_attributes(model)
    assert model.discard, "model has no clusters"
    if model.compress or model.retain:
        sys.stderr.write("Warning, you are predicting on a non finalized model."
                         " Expect less accuracy")


def check_points(points, model, required_nr):
    assert isinstance(points, numpy.ndarray), "Input points not numpy.ndarray"
    rows, dimensions = numpy.shape(points)
    assert dimensions == model.dimensions, "Dimension of points do not match model.dimensions"
    assert rows >= required_nr, "Not enough points"
