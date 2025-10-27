from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import list_images, extract_features
import joblib
import matplotlib.pyplot as plt


def run(data_dir='./data', n_clusters=2, feat_method='hog'):
    paths, labels = list_images(data_dir)
    X = extract_features(paths, method=feat_method)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    preds = kmeans.fit_predict(X)
    score = silhouette_score(X, preds)

    print("Silhouette Score:", score)
    joblib.dump(kmeans, "kmeans_model.pkl")

    plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='viridis', s=5)
    plt.title("K-Means Clustering Result")
    plt.savefig("kmeans_clusters.png")
    plt.show()


if __name__ == "__main__":
    run()
