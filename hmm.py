import os
import numpy as np
from hmmlearn import hmm
from utils import list_images, extract_features
from collections import defaultdict
import joblib


def build_sequences_by_patient(paths, patient_id_fn, feat_method='hog'):
    groups = defaultdict(list)
    for p in paths:
        pid = patient_id_fn(p)
        groups[pid].append(p)

    sequences = []
    for pid, ps in groups.items():
        ps = sorted(ps)
        feats = extract_features(ps, method=feat_method)
        sequences.append(feats)
    return sequences


def run(data_dir='./data', n_states=4):
    paths, labels = list_images(data_dir)

    def pid_from_path(p):
        return p.split(os.sep)[-2]  # folder name as patient ID

    seqs = build_sequences_by_patient(paths, pid_from_path)
    X = np.vstack(seqs)
    lengths = [s.shape[0] for s in seqs]

    model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100)
    model.fit(X, lengths)

    joblib.dump(model, "hmm_model.pkl")
    print("HMM model trained and saved.")


if __name__ == "__main__":
    run()
