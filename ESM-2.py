import datasets, torch, pickle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time

dataset_name = "tattabio/ec_classification"
model_id = ["facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm2_t36_3B_UR50D",
            "facebook/esm2_t48_15B_UR50D"]
out_file = ["data/ESM2_8.pkl",
            "data/ESM2_35.pkl",
            "data/ESM2_150.pkl",
            "data/ESM2_650.pkl",
            "data/ESM2_3B.pkl",
            "data/ESM2_15B.pkl",]
bs = 1
max_len = 5120

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
print("Device:", device, "dtype:", dtype)

def load_data(id):
    ds = datasets.load_dataset(id, trust_remote_code=True)
    labels = set()
    for split in ds:
        labels |= set(ds[split]["Label"])
    return ds, {l:i for i,l in enumerate(sorted(labels))}

def get_model(model, dt):
    tokenizer = AutoTokenizer.from_pretrained(model, 
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(model, 
                                      torch_dtype=dt, 
                                      trust_remote_code=True)
    return model.to(device).eval(), tokenizer

def embed(model, tokenizer, seqs):
    enc = tokenizer(seqs, 
                    truncation=True, 
                    padding="longest", 
                    return_tensors="pt", 
                    max_length=max_len).to(device)
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)
    with torch.no_grad(): 
        h = model(**enc).last_hidden_state
    me = mask.unsqueeze(-1).expand_as(h).float()
    ret = (h * me).sum(1) / (me.sum(1) + 1e-9)
    return ret.cpu()

def proc(ds, model, tokenizer, mp):
    out = {}
    for split in ("train","test"):
        if split not in ds: continue
        embs, labs = [], []
        data = ds[split]
        for i in tqdm(range(0, len(data), bs)):
            b = data.select(range(i, min(i+bs, len(data))))
            seqs = b["Sequence"] if isinstance(b["Sequence"], list) else [b["Sequence"]]
            lbls = b["Label"] if isinstance(b["Label"], list) else [b["Label"]]
            e = embed(model, tokenizer, seqs)
            embs.append(e)
            labs += [mp[l] for l in lbls]
        out[split] = (torch.cat(embs), torch.tensor(labs, dtype=torch.long))
    return out

def main():
    for i in range(5,6):
        print(model_id[i])
        ds, mp = load_data(dataset_name)
        model, tokenizer = get_model(model_id[i], dtype)
        res = proc(ds, model, tokenizer, mp)
        del model,tokenizer
        time.sleep(20)
        data = {}
        for s,(e,l) in res.items():
            data[f"{s}_embeddings"] = e.to(torch.float32).numpy()
            data[f"{s}_labels"] = l.numpy()
        with open(out_file[i],'wb') as f:
            pickle.dump(data,f)

if __name__=="__main__":
    main()