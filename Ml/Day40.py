from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
iris=(load_iris(as_frame=True))

df=iris.frame

# print(df.head())
X=iris.data
y=iris.target
#scaling the features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# print(X_scaled)
# sb.scatterplot(x=X_scaled[:,1],y=X_scaled[:,2])
# plt.show()

Z=linkage(
    X_scaled,
    method="ward"
)

# dendrogram(Z)
# plt.show()
#suggestive as 2 

agg=AgglomerativeClustering(n_clusters=3)
labels=agg.fit_predict(X_scaled)
sb.scatterplot(x=X_scaled[:,0],y=X_scaled[:,2],c=labels)
plt.show()