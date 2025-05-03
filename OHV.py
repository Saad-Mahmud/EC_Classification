import datasets
import numpy as np
import pickle

dataset_name = "tattabio/ec_classification"
out_file = "data/OHV.pkl"
ALPHABET = "-ACDEFGHIKLMNPQRSTVWYBZXOU"
char2int = {c:i for i,c in enumerate(ALPHABET)}
char2idx = {c:i for i,c in enumerate(ALPHABET)}
A = len(ALPHABET)

ds = datasets.load_dataset(dataset_name)
splits = ("train", "test")
seqs = {s: ds[s]["Sequence"] for s in splits}
labs = {s: ds[s]["Label"]    for s in splits}

all_labels = set(labs["train"]) | set(labs["test"])
label2id = {l:i for i,l in enumerate(sorted(all_labels))}

max_len = max(len(s) for split in splits for s in seqs[split])

def one_hot_flat(strings):
    n = len(strings)
    M = np.zeros((n, max_len, A), dtype=np.int8)
    for i, s in enumerate(strings):
        for j, c in enumerate(s.upper()[:max_len]):
            idx = char2idx.get(c)
            if idx is not None:
                M[i, j, idx] = 1
    return M.reshape(n, max_len * A)

X_train = one_hot_flat(seqs["train"])
y_train = np.array([label2id[l] for l in labs["train"]], dtype=np.int64)
X_test  = one_hot_flat(seqs["test"])
y_test  = np.array([label2id[l] for l in labs["test"]],  dtype=np.int64)

with open(out_file, "wb") as f:
    pickle.dump({
        "train_embeddings": X_train,
        "train_labels":     y_train,
        "test_embeddings":  X_test,
        "test_labels":      y_test
    }, f)
