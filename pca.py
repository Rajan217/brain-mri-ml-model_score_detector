from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import list_images, extract_features


def run(data_dir='./data', feat_method='flatten', n_components=50):
    paths, labels = list_images(data_dir)
    X = extract_features(paths, method=feat_method)

    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)

    print("Explained variance ratio (sum):", pca.explained_variance_ratio_.sum())

    if Xp.shape[1] >= 2:
        plt.scatter(Xp[:, 0], Xp[:, 1], s=5)
        plt.title("PCA 2D Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig("pca_2d.png")
        plt.show()


if __name__ == "__main__":
    run()
