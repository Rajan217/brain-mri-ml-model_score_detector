from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import list_images, extract_features, train_test_split_paths
import joblib


def run(data_dir='./data'):
    paths, labels = list_images(data_dir)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split_paths(paths, labels)

    X_train = extract_features(X_train_paths, method='hog')
    X_test = extract_features(X_test_paths, method='hog')

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    bag = BaggingClassifier(estimator=rf, n_estimators=10, random_state=0)
    vote = VotingClassifier(estimators=[('rf', rf), ('bag', bag)], voting='soft')

    vote.fit(X_train, y_train)
    preds = vote.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(vote, "ensemble_model.pkl")


if __name__ == "__main__":
    run()
