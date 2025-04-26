import datasets
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pickle
import sys 

# --- Configuration ---
dataset_name = "tattabio/ec_classification"
model_id = "tattabio/gLM2_650M_embed"
model_name = "gLM2_650M_embed"
output_pickle_file = f"{model_name}_embeddings.pkl"
batch_size = 2
max_length = 8196

# --- Device and Dtype Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
print(f"Using device: {device}, Model dtype: {model_dtype}")

def load_data_and_map_labels(dataset_name: str) -> Optional[Tuple[datasets.DatasetDict, Dict[str, int]]]:
    """Loads dataset and creates label mapping."""
    dataset_dict = datasets.load_dataset(dataset_name)
    unique_labels = set(dataset_dict['train']['Label']) | set(dataset_dict['test']['Label'])
    label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
    print(f"Dataset loaded. Found {len(label_to_id)} unique labels.")
    return dataset_dict, label_to_id

def get_model_and_tokenizer(model_id: str) -> Optional[Tuple[AutoModel, AutoTokenizer]]:
    """Loads model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=model_dtype, trust_remote_code=True).to(device)
    return model, tokenizer

def get_embeddings(model: AutoModel, tokenizer: AutoTokenizer, sequences: List[str]) -> torch.Tensor:
    """Extracts embeddings (assumes pooler_output)."""
    sequences_with_prefix = ["<+>" + seq for seq in sequences]
    batch_encoding = tokenizer(sequences_with_prefix, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
    input_ids = batch_encoding.input_ids.to(device)
    attention_mask = batch_encoding.attention_mask.to(device)
    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    # Assumes pooler_output exists and is correct for this model
    return model_output.pooler_output

def process_splits(dataset_dict: datasets.DatasetDict, model: AutoModel, tokenizer: AutoTokenizer, label_to_id: Dict[str, int]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Processes train and test splits to get embeddings and labels."""
    results = {}
    for split_name in ['train', 'test']:
        print(f"Processing split: {split_name}...")
        current_dataset_split = dataset_dict[split_name]
        all_embeddings = []
        all_labels_indices = []
        num_examples = len(current_dataset_split)

        for i in range(0, num_examples, batch_size):
            batch_data = current_dataset_split[i:min(i + batch_size, num_examples)]
            batch_sequences = batch_data['Sequence']
            batch_string_labels = batch_data['Label']
            if not isinstance(batch_sequences, list): # Handle last batch if size 1
                 batch_sequences = [batch_sequences]
                 batch_string_labels = [batch_string_labels]

            embeddings = get_embeddings(model, tokenizer, batch_sequences)
            if embeddings is not None:
                all_embeddings.append(embeddings.cpu())
                all_labels_indices.extend([label_to_id[lbl] for lbl in batch_string_labels]) # Assumes all labels are known
            # No explicit handling for embedding errors here for brevity

        if all_embeddings and all_labels_indices:
            embeddings_tensor = torch.cat(all_embeddings, dim=0)
            labels_tensor = torch.tensor(all_labels_indices, dtype=torch.long)
            if embeddings_tensor.shape[0] == labels_tensor.shape[0]:
                 results[split_name] = (embeddings_tensor, labels_tensor)
            else:
                 print(f"Warning: Mismatch in {split_name} shapes. Skipping split.")
        else:
             print(f"Warning: No results for {split_name} split.")
    return results

def main():
    """Main execution flow."""
    dataset_dict, label_to_id = load_data_and_map_labels(dataset_name)
    model, tokenizer  = get_model_and_tokenizer(model_id)
    split_results = process_splits(dataset_dict, model, tokenizer, label_to_id)
    

    del model, tokenizer
    if device.type == 'cuda': torch.cuda.empty_cache()

    
    final_data_to_save = {
        'train_embeddings': split_results['train'][0].to(torch.float32).numpy(),
        'train_labels': split_results['train'][1].numpy(),
        'test_embeddings': split_results['test'][0].to(torch.float32).numpy(),
        'test_labels': split_results['test'][1].numpy(),
    }

    with open(output_pickle_file, 'wb') as f:
        pickle.dump(final_data_to_save, f)
    print(f"Successfully saved embeddings and labels to {output_pickle_file}")
    
   
if __name__ == "__main__":
    main()
