import pickle, os, sys
import numpy as np

INPUTS = ["data/KM_pca.pkl", "data/IEV_pca.pkl"]
OUTPUT = "data/KM+IEV.pkl"

def load(f):
    d = pickle.load(open(f, "rb"))
    return d["train_embeddings"], d["train_labels"], d["test_embeddings"], d["test_labels"]

X_tr, y_tr, X_te, y_te = load(INPUTS[0])
print(f"Ref shapes: {X_tr.shape}, {X_te.shape}")

for f in INPUTS[1:]:
    Xt2, yt2, Xe2, ye2 = load(f)
    assert y_tr.shape == yt2.shape == y_tr.shape, "Label mismatch"
    assert X_tr.shape[0] == Xt2.shape[0], "Train count mismatch"
    assert X_te.shape[0] == Xe2.shape[0], "Test count mismatch"
    X_tr = np.hstack([X_tr, Xt2])
    X_te = np.hstack([X_te, Xe2])
    print(f"Added from {os.path.basename(f)} → new shapes: {X_tr.shape}, {X_te.shape}")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
pickle.dump({
    "train_embeddings": X_tr,
    "train_labels":     y_tr,
    "test_embeddings":  X_te,
    "test_labels":      y_te
}, open(OUTPUT, "wb"))

