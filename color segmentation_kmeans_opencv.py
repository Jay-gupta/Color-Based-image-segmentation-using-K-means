import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import cv2


img = cv2.imread("C:\Python36\doodles.jpg")
cv2.imshow("original image",img)

img1 = np.reshape(img, (img.shape[0]*img.shape[1], 3))

cl = cluster.KMeans(n_clusters =4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

d = cl.fit(img1)

y = d.predict(img1)

print(y)

##plt.scatter(img1[:,0], img1[:,1], c=y)
##plt.grid()
##plt.xlabel("attributes")
##plt.ylabel("clusters")
##
##plt.show(5000)


m = np.where(y==2)
img1[m, 0] = 0
#img1[m, 1] = 0
img1[m, 2] = 0

img2 = np.reshape(img1,(img.shape[0], img.shape[1], 3))

cv2.imshow("changed image",img2)
#img2 = cv2.imwrite("C:\Python36\doodles2.jpg")
