""" This module defines the built in way of plotting a bfr model"""

import numpy
import matplotlib.pyplot
import matplotlib.lines
from mpl_toolkits.mplot3d import Axes3D

from . import clustlib


class BfrPlot:
    """ A bfr model

    Attributes
    ----------
    model : bfr.Model
        The model to plot.

    dimensions : int
        The dimensionality of the plot. 2 or 3.

    points : numpy.ndarray
        A BfrPlot can optionally add points to the plot.
        Points are a (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    axis : matplotlib.artist.Artist.axes
        The axes with either 2d or 3d projection

    cmap: matplotlib.pyplot.cm
        The colormap of the plot

    """

    def __init__(self, model, points=None, outlier_detection=True):
        self.model = model
        self.dimensions = model.dimensions
        self.points = points
        if points is not None:
            self.predictions = model.predict(points, outlier_detection)
        if self.dimensions == 3:
            self.axis = create_axis(projection="3d")
        else:
            self.axis = create_axis()
        self.cmap = matplotlib.pyplot.cm.get_cmap()

    def show(self):
        """ Shows the plot.

        Returns
        -------

        """

        legend_entries = []
        for idx in range(-1, len(self.model.discard)):
            if idx == -1:
                col = "black"
            else:
                col = self.cmap(idx)
            if self.points is not None:
                corr_points = find_points(self.points, self.predictions, idx)
                cords = corr_points.T
                colors = [col for each in corr_points]
                if self.dimensions == 3:
                    self.axis.scatter(cords[0], cords[1], cords[2], c=colors)
                else:
                    self.axis.scatter(cords[0], cords[1], c=colors)
            if idx == -1:
                continue
            add_legend_entry(legend_entries, col)
            shape = get_cluster_shape(self.model, self.model.discard[idx])
            label = str(idx) + "\n" + str(self.model.discard[idx])
            if self.dimensions == 3:
                self.axis.plot_surface(shape[0], shape[1], shape[2],
                                       color=col, alpha=0.2, label=label)
            else:
                self.axis.fill(shape[0], shape[1], color=col, alpha=0.2, label=label)
        _, labels = self.axis.get_legend_handles_labels()
        self.axis.legend(legend_entries, labels, loc=1)
        matplotlib.pyplot.show()
        matplotlib.pyplot.close("all")


def create_axis(projection=None):
    """ Creates a 2 or 3 dimensional axis.

    Parameters
    ----------
    projection : str
        The returned axes will be 3d if projection is "3d". 2d otherwise.

    Returns
    -------
    axis : matplotlib.artist.Artist.axes
        The axes with either 2d or 3d projection

    """

    fig = matplotlib.pyplot.figure()
    fig.suptitle("Clusters - BFR")
    fig.subplots_adjust(top=0.85)
    matplotlib.pyplot.set_cmap("tab20")
    if projection:
        axis = fig.add_subplot(111, projection=projection)
        axis.set_zlabel("Z")
    else:
        axis = fig.add_subplot(111)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    return axis


def add_legend_entry(legend_entries, col):
    """ The 3d projection of a matplotlib scatterplot does not support a legend.
    This function is a workaround which allows a legend to be shown in 3d projection.

    Parameters
    ----------
    legend_entries : list
        A list of invisible matplotlib proxy objects.

    col : keyword or RGBA
        see matplotlib.colors

    Returns
    -------

    """

    entry_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c=col, marker="H")
    legend_entries.append(entry_proxy)


def find_points(points, predictions, cluster_idx):
    """ Finds the points which are predicted to a given cluster index.

    Parameters
    ----------
    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    predictions : numpy.ndarray
        The indices of the closest cluster of each point (defined by default distance_fun).
        Returns -1 if the point is considered an outlier (determined by
        default threshold_fun and threshold)

    cluster_idx : int
        The cluster index of which predictions will be identified.

    Returns
    -------
    corresponding points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points predicted to cluster_idx.

    """

    corr_idx = numpy.where(predictions == cluster_idx)
    return points[corr_idx]


def get_cluster_shape(model, cluster):
    """ Computes the shape of cluster by the default threshold of model.
    The shape is computed by parametrizing the dimensions using circular or
    spherical coordinates.

    Parameters
    ----------
    model : bfr.Model
        The model specifies the shape by its default threshold.

    cluster : bfr.clustlib.Cluster

    Returns
    -------
    cords : numpy.ndarray
        Matrix with points on the envelope surface of the cluster.

    """

    mean = clustlib.mean(cluster)
    radius = confidence_interval(cluster, model.threshold)
    resolution = 20
    if model.dimensions == 2:
        resolution *= 10
    u_sub = numpy.linspace(0, 2 * numpy.pi, resolution)
    v_sub = numpy.linspace(0, numpy.pi, resolution)
    if model.dimensions == 3:
        x_cord = radius[0] * numpy.outer(numpy.cos(u_sub), numpy.sin(v_sub)) + mean[0]
        y_cord = radius[1] * numpy.outer(numpy.sin(u_sub), numpy.sin(v_sub)) + mean[1]
        all_ones = numpy.ones(numpy.size(u_sub))
        z_cord = radius[2] * numpy.outer(all_ones, numpy.cos(v_sub)) + mean[2]
        cords = (x_cord, y_cord, z_cord)
    else:
        x_cord = radius[0] * numpy.cos(u_sub) + mean[0]
        y_cord = radius[1] * numpy.sin(u_sub) + mean[1]
        cords = (x_cord, y_cord)
    return cords


def confidence_interval(cluster, threshold):
    """ Computes cluster shape based on the cluster and threshold.
        If mahalanobis threshold is set to N * sqrt(dimensions)
        the shape will correspond to a confidence interval equal to
        N standard deviations of a normal distribution.

    Parameters
    ----------
    cluster : bfr.clustlib.Cluster

    threshold : float


    Returns
    -------
    shape : float, float, float
        the radius in each dimension

    """

    dimensions = len(cluster.sums)
    distance = threshold ** 2
    std_dev = clustlib.std_dev(cluster)
    width = numpy.sqrt(distance * std_dev[0] ** 2)
    height = numpy.sqrt(distance * std_dev[1] ** 2)
    if dimensions == 2:
        return width, height
    breadth = numpy.sqrt(distance * std_dev[2] ** 2)
    return width, height, breadth
