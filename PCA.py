import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

IN_PKL  = ["data/OHV.pkl","data/IEV.pkl", "data/KM.pkl"]
OUT_PKL = ["data/OHV_pca.pkl","data/IEV_pca.pkl", "data/KM_pca.pkl"]
N_COMP  = 512

for i in range(len(IN_PKL)):
    with open(IN_PKL[i], "rb") as f:
        d = pickle.load(f)
    X_tr, y_tr = d["train_embeddings"], d["train_labels"]
    X_te, y_te = d["test_embeddings"],  d["test_labels"]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    pca = PCA(n_components=N_COMP, random_state=42).fit(X_tr_s)
    X_tr_p = pca.transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)

    with open(OUT_PKL[i], "wb") as f:
        pickle.dump({
            "train_embeddings": X_tr_p,
            "train_labels":     y_tr,
            "test_embeddings":  X_te_p,
            "test_labels":      y_te
        }, f)
