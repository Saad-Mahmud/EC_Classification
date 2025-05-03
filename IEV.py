import datasets
import numpy as np
import pickle

dataset_name = "tattabio/ec_classification"
out_file = "data/IEV.pkl"
ALPHABET = "-ACDEFGHIKLMNPQRSTVWYBZXOU"
char2int = {c:i for i,c in enumerate(ALPHABET)}

ds = datasets.load_dataset(dataset_name)
splits = ("train", "test")
seqs = {s: ds[s]["Sequence"] for s in splits}
labs = {s: ds[s]["Label"]    for s in splits}

all_labels = set(labs["train"]) | set(labs["test"])
label2id = {l:i for i,l in enumerate(sorted(all_labels))}

max_len = max(len(s) for split in splits for s in seqs[split])

def encode_list(strings):
    arr = np.zeros((len(strings), max_len), dtype=np.int32)
    for i, s in enumerate(strings):
        for j, c in enumerate(s.upper()):
            arr[i,j] = char2int.get(c, char2int["-"])
    return arr

X_train = encode_list(seqs["train"])
y_train = np.array([label2id[l] for l in labs["train"]], dtype=np.int64)
X_test  = encode_list(seqs["test"])
y_test  = np.array([label2id[l] for l in labs["test"]],  dtype=np.int64)

with open(out_file, "wb") as f:
    pickle.dump({
        "train_embeddings": X_train,
        "train_labels":     y_train,
        "test_embeddings":  X_test,
        "test_labels":      y_test
    }, f)
