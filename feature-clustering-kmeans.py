import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive',force_remount=True)

# Load the data
data = pd.read_csv('/content/drive/MyDrive/data/training_data/combined_set.csv')

# Drop non-feature columns
features = data.drop(columns=['gene', 'dependent'])

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Perform PCA to reduce to two dimensions for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)

# Determine the inertia for a range of k values (elbow method)
inertia = []
k_values = range(1, 11)  # Adjust this range as needed
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_features)
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Prompt the user to enter the optimal number of clusters
#optimal_k = int(input("Enter the optimal number of clusters (from the elbow plot): "))

optimal_k = 3

# Perform k-means clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(reduced_features)

# Plotting the clustered data
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', marker='o', alpha=0.6)
plt.title('Clustered Data Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Add cluster labels to the original data and save to a new CSV
data['cluster'] = clusters
data.to_csv('/content/drive/MyDrive/data/training_data/feature_clusters_kmeans.csv', index=False)

from sklearn.metrics import silhouette_score

# Assuming you have found the optimal_k from the elbow plot and have the reduced_features
optimal_k = int(input("Enter the optimal number of clusters (from the elbow plot): "))
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# Calculate the silhouette score
silhouette_avg = silhouette_score(reduced_features, cluster_labels)
print(f"The average silhouette_score is : {silhouette_avg}")

# Assuming `data` is your original DataFrame and it now includes a 'cluster' column
# from k-means and the original 'dependent' column with known labels.

# Group by the known label and get counts of each cluster label within each group
cross_tab = pd.crosstab(data['dependent'], data['cluster'])

# Calculate percentages of each cluster within each known label group
cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0)

# Display the cross-tabulation
print(cross_tab)
print(cross_tab_percentage)

# You can also visualize this using a bar chart
cross_tab_percentage.plot(kind='bar', stacked=True)
plt.title('Cross-Reference of Clusters with Known Labels')
plt.xlabel('Known Label')
plt.ylabel('Percentage of Data Points in Each Cluster')
plt.show()

