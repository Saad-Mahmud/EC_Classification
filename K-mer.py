import datasets, numpy as np, pickle
from itertools import product

dataset_name = "tattabio/ec_classification"
out_file = "data/KM.pkl"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
Ks  = (1, 2, 3, 4)

kmap = {
            k: { "".join(p): i for i, p in enumerate(product(ALPHABET, repeat=k)) } 
            for k in Ks
        }

ds     = datasets.load_dataset(dataset_name)
splits = ("train", "test")
seqs = {s: ds[s]["Sequence"] for s in splits}
labs = {s: ds[s]["Label"]    for s in splits}

all_labels = set(labs["train"]) | set(labs["test"])
label2id = {l:i for i,l in enumerate(sorted(all_labels))}

def fk(seq):
    s = "".join(c for c in seq.upper() if c in ALPHABET)
    parts = []
    for k in Ks:
        m    = kmap[k]
        cnt  = np.zeros(len(m), np.float32)
        n    = len(s) - k + 1
        for i in range(max(n, 0)):
            cnt[m[s[i:i+k]]] += 1
        parts.append(cnt / (n if n > 0 else 1))
    return np.concatenate(parts)

out = {}
for split in splits:
    S  = ds[split]
    X  = np.vstack([fk(seq) for seq in S["Sequence"]])
    y  = np.array([label2id[l] for l in S["Label"]], np.int64)
    out[f"{split}_embeddings"] = X
    out[f"{split}_labels"]     = y

with open(out_file, "wb") as f:
    pickle.dump(out, f)
