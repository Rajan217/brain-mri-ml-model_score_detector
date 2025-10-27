from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from utils import list_images, extract_features
import joblib


def run(data_dir='./data', n_components=2, feat_method='hog'):
    paths, labels = list_images(data_dir)
    X = extract_features(paths, method=feat_method)

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    preds = gmm.fit_predict(X)

    print("BIC:", gmm.bic(X))
    print("Silhouette Score:", silhouette_score(X, preds))
    joblib.dump(gmm, "gmm_model.pkl")


if __name__ == "__main__":
    run()
