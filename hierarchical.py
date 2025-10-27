from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from utils import list_images, extract_features


def run(data_dir='./data', n_clusters=2, feat_method='hog'):
    paths, labels = list_images(data_dir)
    X = extract_features(paths, method=feat_method)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    preds = model.fit_predict(X)

    print("Silhouette Score:", silhouette_score(X, preds))


if __name__ == "__main__":
    run()
