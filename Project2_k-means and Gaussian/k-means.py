print(__doc__)
from sklearn import preprocessing
from sklearn.mixture import GMM

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(10, 10))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(331)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# Anisotropicly distributed data
transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(332)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(333)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

plt.subplot(334)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

'''
# Invrease x-feature by 5
X_increased = np.copy(X)
for a in X_increased:
    a[0] = 5 * a[0]
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_increased)

plt.subplot(335)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Invrease x-feature by 5")
'''

# normalized data
transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(np.copy(X), transformation)
X_nor = preprocessing.normalize((X_aniso), norm='l2')
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_nor)

plt.subplot(335)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("normalized datas")

# GMM incorrect number of clusters
X_increased = np.copy(X)
y_pred = GMM(n_components=2, random_state=random_state).fit_predict(X_increased)

plt.subplot(336)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("GMM_incorrect number of clusters")

# GMM2 anisotropicly distributed blobs
#transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
#X_aniso = np.dot(np.copy(X), transformation)
y_pred = GMM(n_components=3, covariance_type='tied', random_state=random_state).fit_predict(X_aniso)

plt.subplot(337)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("GMM_anisotropicly distributed blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = GMM(n_components=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(338)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("GMM_Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = GMM(n_components=3, random_state=random_state).fit_predict(X_filtered)

plt.subplot(339)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("GMM_Unevenly Sized Blobs")

plt.show()