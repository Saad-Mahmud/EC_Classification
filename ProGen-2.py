import datasets, torch, pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time

dataset_name = "tattabio/ec_classification"
model_id = ["hugohrban/progen2-xlarge", #6.44B
            "hugohrban/progen2-large", #2.78B
            "hugohrban/progen2-medium", #765M
            "hugohrban/progen2-small",] #151M
out_file = ["data/PRG2_x.pkl","data/PRG2_l.pkl","data/PRG2_m.pkl","data/PRG2_s.pkl"]
bs = 1
max_len = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = AutoModelForCausalLM.from_pretrained(model, 
                                      torch_dtype=dt, 
                                      trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<PAD>"
    return model.to(device).eval(), tokenizer

def embed(model, tokenizer, seqs):
    
    enc = tokenizer(seqs, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt", 
                    max_length=max_len).to(device)
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)
    with torch.no_grad():
        h = model(input_ids=ids, attention_mask=mask,
                output_hidden_states=True).hidden_states[-1]
    me = mask.unsqueeze(-1).expand_as(h).float()
    ret = (h * me).sum(1) / me.sum(1).clamp(min=1e-9)
    return ret.cpu()

def proc(ds, model, tokenizer, mp):
    out = {}
    al = []
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
    for i in range(4):
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