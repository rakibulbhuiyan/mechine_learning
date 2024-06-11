import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
dataset=pd.read_csv('F:\git\Mechine_learning\k_means\Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values
# print(X)
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=kMeans(n_clusters=5, init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], c='orange',s=100,label='cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], c='green',s=100,label='cluster 1')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], c='black',s=100,label='cluster 1')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], c='yellow',s=100,label='cluster 1')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], c='blue',s=100,label='cluster 1')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c='red',s=300,label='centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()