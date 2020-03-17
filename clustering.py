import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering as cure
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import bfr
import os
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture as expectationMaximization
from sklearn.cluster import DBSCAN  # NO SIRVE
from sklearn.cluster import MeanShift  # NO SIRVE
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import Birch

# Complement Lists used to manage the data
lisstd = []
listDis = []
distK = []
distS = []
acc = {}
# Lists used to plot the TSNE 2 teg
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
markers = ['o', '^', 's', '.', ',', 'x', '+', 'v', 'd', '>']

#path for images just test chamges

path = str(Path(os.getcwd()))


# Method to plot the original data with NO CLUSTERS
def tsneG(X):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    dat = pd.DataFrame()
    dat['x1'] = X_2d[:, 0]
    dat['y1'] = X_2d[:, 1]

    plt.scatter(dat["x1"], dat["y1"])
    plt.title("Raw data with no clusters")
    plt.grid()
    fig = plt.gcf()
    fig.canvas.set_window_title('Project 7')
    plt.savefig("raw.png")
    plt.close()


# Cure implementation
def Cure(input_data, n):
    cure_instance = cure(n_clusters=n, linkage="average", affinity="euclidean")  # also test with affinity ="cosine"
    cure_instance.fit(input_data)
    pred = cure_instance.labels_
    print(pred)
    print("ACC: " + str(accuracy_score(pred, labels)))
    acc["CURE" + str(n)] = str(accuracy_score(pred, labels))
    tsnePlot(pred, n, input_data, 'CURE')


# EXpectation Mximization implementation
def ExpectationMMaximization(Mat, n):
    exp_instance = expectationMaximization(n_components=n)
    exp_instance.fit(Mat)
    pred = exp_instance.predict(Mat)
    print(pred)
    print("ACC: " + str(accuracy_score(pred, labels)))
    acc["EXP" + str(n)] = str(accuracy_score(pred, labels))
    tsnePlot(pred, n, Mat, 'EXP')


# dbscan implemenation
def dbscan(Mat, n):
    db_instance = DBSCAN(min_samples=n)
    db_instance.fit(Mat)
    pred = db_instance.labels_
    print(pred)
    print("ACC: " + str(accuracy_score(pred, labels)))
    acc["DB" + str(n)] = str(accuracy_score(pred, labels))
    tsnePlot(pred, n, Mat, 'DB')


# Meanshift implemenation
def meanshift(Mat, n):
    mean_instance = MeanShift(seeds=n)
    mean_instance.fit(Mat)
    pred = mean_instance.labels_
    print(pred)
    print("ACC: " + str(accuracy_score(pred, labels)))
    acc["MEAN" + str(n)] = str(accuracy_score(pred, labels))
    tsnePlot(pred, n, Mat, 'MEAN')


# kmeans implemenation
def kmeans(X, n):
    kmeans = KMeans(n_clusters=n, init='k-means++', precompute_distances='auto').fit(X)
    distK.append(kmeans.inertia_)
    pred = kmeans.predict(X)
    print(kmeans.cluster_centers_)
    print(pred)
    print("ACC: \n" + str(accuracy_score(pred, labels)))
    acc['KMeans' + str(n)] = accuracy_score(pred, labels)
    tsnePlot(pred, n, X, 'KMEAN')


# spectarl implementation
def spect(input_data, n):
    spec_instance = SpectralCoclustering(n_clusters=n)
    spec_instance.fit(input_data)
    pred = spec_instance.row_labels_
    print(pred)
    print("ACC: " + str(accuracy_score(pred, labels)))
    acc["SPECT" + str(n)] = str(accuracy_score(pred, labels))
    tsnePlot(pred, n, input_data, 'SPECT')

# birch implementation
def birch(input_data, n):
    feat_instance = Birch(n_clusters=n)
    feat_instance.fit(input_data)
    pred= feat_instance.labels_

    print(pred)
    print("ACC: " + str(accuracy_score(pred, labels)))
    acc["BIRCH" + str(n)] = str(accuracy_score(pred, labels))
    tsnePlot(pred, n, input_data, 'BIRCH ')


# Generate dendoram plot
def dendogram(Mat):
    dendogram = sch.dendrogram(sch.linkage(Mat, method='ward'))
    plt.title("Generated Tree with CURE clustering")
    plt.savefig("CureDend.png")
    plt.show()
    plt.close()


# BFR implementation
# based on the code: https://github.com/jeppeb91/bfr
def BFR(Mat, n, shape):
    model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=3.0,
                      merge_threshold=2.0, dimensions=shape,
                      # Numer of dimensions must be defined as the number of columns of the data
                      init_rounds=40, nof_clusters=n)

    Mat = Mat.to_numpy()
    print(Mat.shape)

    model.fit(Mat)

    model.finalize()
    std = model.error()
    dis = model.error(Mat)  # /598
    print(dis)
    lisstd.append(std)
    listDis.append(dis)
    print("Desviacion estandar del modelo:\n" + str(model.error()))
    print("SSE del modelo:\n" + str(model.error(Mat)))

    centers = pd.DataFrame(model.centers())
    print(centers)
    pred = model.predict(Mat)

    print("ACC: " + str(accuracy_score(pred, labels)))
    acc['BFR' + str(n)] = accuracy_score(pred, labels)
    print(pred)

    print(pred.shape)
    print(model)
    tsnePlot(pred, n, Mat, 'BFR')


''' 
Method that uses TSNE to plot the clustered data
pred -> Labels predicted
n -> number of clusters
Mat-> original matrix of data
alg -> algorith used
'''


def tsnePlot(pred, n, Mat, alg):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(Mat)
    dat = pd.DataFrame()
    dat['x1'] = X_2d[:, 0]
    dat['y1'] = X_2d[:, 1]

    klus = list(range(n))

    for i, c, m in zip(klus, colors, markers):
        plt.scatter(X_2d[pred == i, 0], X_2d[pred == i, 1], marker=m, c=c, label=i)

    plt.savefig(path+"/images/Clusters" + alg + str(n) + ".png")
    # plt.show()
    plt.close()


def graphCurves():
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 10), lisstd, 'o-', label="STD")
    plt.title("Desviacion estandar por # clusters")
    plt.grid()
    plt.ylabel("STD")

    plt.subplot(2, 1, 2)
    plt.plot(range(1, 10), listDis, label="Dist", color='red')
    plt.grid()
    plt.title("Average distance vs  #clusters")
    plt.ylabel("Avg. Distance")
    plt.xlabel("# Clusters")
    plt.savefig("FinalCurves.png")
    plt.close()


def graphK(dist):
    plt.plot(range(1, 10), dist)
    plt.grid()
    plt.title("Average distance vs  #clusters")
    plt.ylabel("Avg. Distance")
    plt.xlabel("# Clusters")
    plt.savefig("FinalCurvesK.png")
    plt.close()


if __name__ == '__main__':

    matrix = pd.read_csv("features_Modified.csv", delimiter=',', header=None)
    labels = pd.read_csv('labels(1VT-0LP).csv', header=None)
    print(path)

    # Starting BFR tests so branch new test
    print("BFR TEST:\n")
    for i in range(2, 11):
        BFR(matrix, i, matrix.shape[1])
    # Start kmenas Test
    print("K MEANS  TEST:\n")
    for i in range(2, 11):
        kmeans(matrix, i)

    # Start Cure test
    print("CURE  TEST:\n")
    for i in range(2, 11):
        Cure(matrix, i)

    # Start Exp. Max test
    print("EXPECTATION MAXIMIZATION  TEST:\n")
    for i in range(2, 11):
        ExpectationMMaximization(matrix, i)
    # Start Spectral test
    print("SPECT  TEST:\n")
    for i in range(2, 11):
        spect(matrix, i)

    # Start Birch test
    print("BIRCH  TEST:\n")
    for i in range(2, 11):
        birch(matrix, i)
    # subprocess.call("python3 Cure.py ", shell=True)

    print(acc)
