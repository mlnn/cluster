import numpy as np
import random
import sklearn.cluster as clstr
import pylab as plt
import matplotlib.cm as cm

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

points = init_board(100)
clusterizator = clstr.MiniBatchKMeans(n_clusters=7, init_size=21)
# clusterizator = clstr.Birch
clusterizator.fit(points, y=None)
p = points.transpose()
plt.scatter(p[0], p[1])
centers = clusterizator.cluster_centers_
c = centers.transpose()
colors = cm.rainbow(np.linspace(0, 1, 7))

for i in range(100):
    label = clusterizator.labels_[i]
    plt.scatter(p[0,i], p[1,i], color=colors[label])

plt.grid()
plt.show()