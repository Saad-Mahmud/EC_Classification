
import datasets, numpy as np, pickle, os
from propy.PyPro import GetProDes          
from tqdm import tqdm

dataset_name   = "tattabio/ec_classification"
out_file   = "data/PROFEAT.pkl"
splits    = ("train", "test")
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")          

ds     = datasets.load_dataset(dataset_name)
splits = ("train", "test")
seqs = {s: ds[s]["Sequence"] for s in splits}
labs = {s: ds[s]["Label"]    for s in splits}

all_labels = set(labs["train"]) | set(labs["test"])
label2id = {l:i for i,l in enumerate(sorted(all_labels))}

def profeat_vec(seq):
    seq_clean = "".join([aa for aa in seq.upper() if aa in ALPHABET])
    desc      = GetProDes(seq_clean)
    subsections = [desc.GetAAComp(), desc.GetDPComp(), desc.GetTPComp(), desc.GetCTD(), 
                   desc.GetMoranAuto(), desc.GetPAAC()]
    return np.fromiter((v for d in subsections for v in d.values()),dtype=np.float32,count=1447)

out = {}
for split in splits:
    print(f"Encoding {split} split …")
    X = np.vstack([profeat_vec(seq) for seq in tqdm(seqs[split])])
    y = np.fromiter((label2id[l] for l in labs[split]), dtype=np.int64)
    out[f"{split}_embeddings"] = X
    out[f"{split}_labels"]     = y

os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "wb") as f:
    pickle.dump(out, f)
