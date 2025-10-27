from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import list_images, extract_features, train_test_split_paths
import joblib


def run(data_dir='./data'):
    paths, labels = list_images(data_dir)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split_paths(paths, labels)

    X_train = extract_features(X_train_paths, method='hog')
    X_test = extract_features(X_test_paths, method='hog')

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(clf, "cart_model.pkl")


if __name__ == "__main__":
    run()
