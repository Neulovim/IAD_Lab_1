import matplotlib.pyplot as plt
import xlrd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs


def read_csv(name: str, index: int):
    rb = xlrd.open_workbook(name)
    sheet = rb.sheet_by_index(index)
    vals = [sheet.row_values(rownum) for rownum in range(sheet.nrows)]
    print(vals)
    return vals


X = np.array(read_csv('lab2.xlsx', 1))
Y = np.array([i for i in range(1, 57)])
# Y = np.array([i for i in range(1, 9)])

# kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
# print(y_kmeans)
# print(kmeans.labels_)

# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(
fig, ([ax11, ax21, ax31, ax41, ax51, ax61, ax71, ax81], (ax12, ax22, ax32, ax42, ax52, ax62, ax72, ax82),
      (ax13, ax23, ax33, ax43, ax53, ax63, ax73, ax83), (ax14, ax24, ax34, ax44, ax54, ax64, ax74, ax84),
      [ax15, ax25, ax35, ax45, ax55, ax65, ax75, ax85], (ax16, ax26, ax36, ax46, ax56, ax66, ax76, ax86),
      (ax17, ax27, ax37, ax47, ax57, ax67, ax77, ax87), (ax18, ax28, ax38, ax48, ax58, ax68, ax78, ax88)) = plt.subplots(
    # nrows=8, ncols=8,
    # figsize=(32, 16)
    nrows=8, ncols=8,
    figsize=(16, 16)
)

# for column1 in range(len(X[0])):
# for column2 in range(len(X[0])):
# ax11.scatter(X[:, 0], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax21.scatter(X[:, 1], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax31.scatter(X[:, 2], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax41.scatter(X[:, 3], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax51.scatter(X[:, 4], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax61.scatter(X[:, 5], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax71.scatter(X[:, 6], X[:, 0], c=y_kmeans, s=50, cmap='viridis')
ax81.scatter(X[:, 7], X[:, 0], c=y_kmeans, s=50, cmap='viridis')

ax12.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# ax22.scatter(X[:, 1], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax32.scatter(X[:, 2], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax42.scatter(X[:, 3], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax52.scatter(X[:, 4], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax62.scatter(X[:, 5], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax72.scatter(X[:, 6], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax82.scatter(X[:, 7], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

ax13.scatter(X[:, 0], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
ax23.scatter(X[:, 1], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
# ax33.scatter(X[:, 2], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
ax43.scatter(X[:, 3], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
ax53.scatter(X[:, 4], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
ax63.scatter(X[:, 5], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
ax73.scatter(X[:, 6], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
ax83.scatter(X[:, 7], X[:, 2], c=y_kmeans, s=50, cmap='viridis')

ax14.scatter(X[:, 0], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax24.scatter(X[:, 1], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax34.scatter(X[:, 2], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
# ax44.scatter(X[:, 3], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax54.scatter(X[:, 4], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax64.scatter(X[:, 5], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax74.scatter(X[:, 6], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax84.scatter(X[:, 7], X[:, 3], c=y_kmeans, s=50, cmap='viridis')

ax15.scatter(X[:, 0], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
ax25.scatter(X[:, 1], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
ax35.scatter(X[:, 2], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
ax45.scatter(X[:, 3], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
# ax55.scatter(X[:, 4], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
ax65.scatter(X[:, 5], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
ax75.scatter(X[:, 6], X[:, 4], c=y_kmeans, s=50, cmap='viridis')
ax85.scatter(X[:, 7], X[:, 4], c=y_kmeans, s=50, cmap='viridis')

ax16.scatter(X[:, 0], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
ax26.scatter(X[:, 1], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
ax36.scatter(X[:, 2], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
ax46.scatter(X[:, 3], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
ax56.scatter(X[:, 4], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
# ax66.scatter(X[:, 5], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
ax76.scatter(X[:, 6], X[:, 5], c=y_kmeans, s=50, cmap='viridis')
ax86.scatter(X[:, 7], X[:, 5], c=y_kmeans, s=50, cmap='viridis')

ax17.scatter(X[:, 0], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
ax27.scatter(X[:, 1], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
ax37.scatter(X[:, 2], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
ax47.scatter(X[:, 3], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
ax57.scatter(X[:, 4], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
ax67.scatter(X[:, 5], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
# ax77.scatter(X[:, 6], X[:, 6], c=y_kmeans, s=50, cmap='viridis')
ax87.scatter(X[:, 7], X[:, 6], c=y_kmeans, s=50, cmap='viridis')

ax18.scatter(X[:, 0], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
ax28.scatter(X[:, 1], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
ax38.scatter(X[:, 2], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
ax48.scatter(X[:, 3], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
ax58.scatter(X[:, 4], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
ax68.scatter(X[:, 5], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
ax78.scatter(X[:, 6], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
# ax88.scatter(X[:, 7], X[:, 7], c=y_kmeans, s=50, cmap='viridis')
"""
for column in range(len(X[0])):
    plt.scatter(Y, X[:, column], c=y_kmeans, s=50, cmap='viridis')
"""
centers = np.array(kmeans.cluster_centers_)
Yc = np.array(([i for i in range(1, len(centers) + 1)]))
print(centers)
# for column in range(len(centers[0])):
#     plt.scatter(Yc, centers[:, column], c='black', s=200, alpha=0.5)

plt.show()


