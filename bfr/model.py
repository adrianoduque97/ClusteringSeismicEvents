""" This is a module defining the interface for bfr"""

import sys
import traceback
import numpy

from . import modellib
from . import objective
from . import setlib
from . import clustlib
from . import error
from . import plot

class Model:
    """ A bfr model

    Attributes
    ----------
    eucl_threshold : float
        Nearness of a point and a cluster is determined by
        Euclidean distance(point,cluster) < eucl_threshold

    merge_threshold : float
        Two clusters in the compress set will be merged if their merged standard deviation
        is less than or equal to (std_dev(cluster) + std_dev(other_cluster)) * merge_threshold.

    init_rounds : int
        Higher integer numbers give better spread of the initial points

    dimensions : int
        The dimensionality of the model

    nof_clusters : int
        The number of clusters (eg. K)

    threshold_fun : function
        The current default function for determining if a point and a cluster
        are considered close. The function should accept a point and a cluster
        and return a float.

    threshold : float
        The current default threshold used by the model. Should point to eucl_threshold when the
        current threshold_fun is Euclidean. Should point to mahal_threshold when the threshold_fun
        is mahalanobis.

    distance_fun : function
        The current default function for finding the closest cluster of a point.
        The function should accept a point and a cluster and return a float

    mahal_threshold : float
        Nearness of point and cluster is determined by
        mahalanobis distance < mahalanobis_factor * sqrt(dimensions)

    discard : list
        The discard set holds all the clusters. A point will update a cluster
        (and thus be discarded) if it is considered near the cluster.

    compress : list
        The compression set holds clusters of points which are near to each other
        but not near enough to be included in a cluster of the discard set

    retain : list
        Contains uncompressed outliers which are neither near to other points nor a cluster

    initialized : bool
        True if initialization phase has been completed, false otherwise.

    """

    def __init__(self, **kwargs):
        self.eucl_threshold = kwargs.pop('euclidean_threshold', "error")
        self.merge_threshold = kwargs.pop('merge_threshold', "error")
        self.init_rounds = kwargs.pop('init_rounds', 5)
        self.dimensions = kwargs.pop('dimensions', "error")
        self.nof_clusters = kwargs.pop('nof_clusters', "error")
        mahal_factor = kwargs.pop('mahalanobis_factor', 3.0)

        self.threshold_fun = clustlib.euclidean
        self.threshold = self.eucl_threshold
        self.distance_fun = clustlib.euclidean

        self.mahal_threshold = mahal_factor * numpy.sqrt(self.dimensions)

        self.discard = []
        self.compress = []
        self.retain = []
        self.initialized = False

        error.confirm_attributes(self)

        # Floating point errors result in sqrt of negative nrs in rare cases.
        # The NaN nrs are treated correctly and warnings may thus be ignored.
        numpy.seterr(invalid='ignore')

    def fit(self, input_points, initial_points=None):
        """ Fits a bfr model with input_points optionally using
        the initial points specified in initial points.

        Parameters
        ----------
        input_points : numpy.ndarray
            (rows, dimensions) array with rows consisting of points. The points should
            have the same dimensionality as the model.

        initial_points : numpy.ndarray
            Array with rows of points that will be used as the initial centers.
            The points should have the same dimensionality as the model
            and the number of points should be equal to the number of clusters.

        Returns
        -------

        """

        next_idx = 0
        if not self.initialized:

            error.confirm_initial_fit(input_points, self)

            input_points = numpy.copy(input_points)
            next_idx = modellib.initialize(input_points, self, initial_points)
            if not next_idx:
                return
            modellib.enable_mahalanobis(self)

        try:
            error.confirm_initialized_fit(input_points, self)
        except AssertionError:
            traceback.print_exc()
            return

        modellib.cluster_points(input_points[next_idx:], self, objective.finish_points)
        setlib.update_compress(self)

    def finalize(self):
        """ Forces the model to assign all clusters in the compress and retain set to
        their closest center in discard. For best results, call this when the model has
        been created and (potentially) updated with all points.

        Returns
        -------

        """

        try:
            error.confirm_attributes(self)
        except AssertionError:
            traceback.print_exc()
            return

        setlib.finalize_set(self.compress, self)
        setlib.finalize_set(self.retain, self)
        self.compress = []
        self.retain = []
        if objective.zerofree_variances(model=self):
            modellib.enable_mahalanobis(self)

    def predict(self, points, outlier_detection=False):
        """ Predicts which cluster a point belongs to.

        Parameters
        ----------
        points : numpy.ndarray
            (rows, dimensions) array with rows consisting of points. The points should
            have the same dimensionality as the model.

        outlier_detection : bool
            If True, outliers will be predicted with -1.
            If False, predictions will not consider default threshold.

        Returns
        -------
        predictions : numpy.ndarray
            The index of the closest cluster (defined by default distance_fun).
            Returns -1 if the point is considered an outlier (determined by
            default threshold_fun and threshold)

        """

        try:
            error.confirm_predict(points, self)
        except AssertionError:
            #traceback.print_exc()

            return None

        if not self.initialized:
            sys.stderr.write(".\n")

        nof_predictions = len(points)
        predictions = numpy.zeros(nof_predictions)
        for idx in range(nof_predictions):
            point = points[idx]
            predictions[idx] = modellib.predict_point(point, self, outlier_detection)
        return predictions.astype(int)

    def error(self, points=None, outlier_detection=False):
        """ Computes the error of the model measured with points.

        Parameters
        ----------
        points : numpy.ndarray
            (rows, dimensions) array with rows consisting of points. The points should
            have the same dimensionality as the model.

        outlier_detection : bool
            If True, outliers will be ignored when computing the error.
            If False, all points will be considered.


        Returns
        -------
        Residual sum of squares : float
            A rate of how far all points are from their closest cluster centers.

        """

        try:
            error.confirm_error(points, self)
        except AssertionError:
            #traceback.print_exc()
            #print("Little prob")
            return 0
        if points is None:
            return modellib.std_error(self)
        return modellib.rss_error(points, self, outlier_detection)

    def centers(self):
        """ Returns all cluster centers.

        Returns
        -------
        means : numpy.ndarray
            Rows correspond to centers of all clusters in the discard set

        """

        try:
            error.confirm_centers(self)
        except AssertionError:
            print(AssertionError)

        means = numpy.zeros((self.nof_clusters, self.dimensions))
        for idx, cluster in enumerate(self.discard):
            means[idx] = clustlib.mean(cluster)
        return means

    def plot(self, points=None, outlier_detection=True):
        """ Plot the model. The dimensions of clusters are represented by default threshold.
        If mahalanobis threshold is set to N * sqrt(dimensions)
        the shape will correspond to a confidence interval equal to
        N standard deviations of a normal distribution.

        Parameters
        ----------
        points : numpy.ndarray
            Optionally add points to the plot.
            Points are a (rows, dimensions) array with rows consisting of points.
            The points should have the same dimensionality as the model.

        outlier_detection : bool
            If points are provided and outlier detection = True,
            outliers will be identified and plotted as black dots.
            If False, all points will be assigned to their closest cluster.

        Returns
        -------

        """

        try:
            error.confirm_plot(points, self)
        except AssertionError:
            #traceback.print_exc()

            return 0

        bfr_plot = plot.BfrPlot(self, points, outlier_detection)
        bfr_plot.show()

    def __str__(self):
        res = ""
        for idx, cluster in enumerate(self.discard):
            res += str(idx) + "\n" + str(cluster) + "\n"
        return res
