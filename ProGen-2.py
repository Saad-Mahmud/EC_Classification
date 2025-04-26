import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # Changed import
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pickle
import sys
import gc # For garbage collection if needed

# --- Configuration ---
dataset_name = "tattabio/ec_classification"
# Model ID for ProGen2 Large
model_id = "hugohrban/progen2-xlarge"
model_name = "progen2-large" # Simplified name for output file
output_pickle_file = f"{model_name}_embeddings_{dataset_name.split('/')[-1]}.pkl" # More descriptive filename
batch_size = 2 # Keep batch size small for large models, adjust based on GPU memory
# Max sequence length for ProGen2 Large
max_length = 1024

# --- Device and Dtype Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use "auto" or torch.bfloat16 if supported for large models
model_dtype = "auto" # Let transformers handle dtype selection (e.g., bfloat16 if available)
print(f"Using device: {device}, Requested Model dtype: {model_dtype}")

def load_data_and_map_labels(dataset_name: str) -> Optional[Tuple[datasets.DatasetDict, Dict[str, int]]]:
    """Loads dataset and creates label mapping."""
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset_dict = datasets.load_dataset(dataset_name)
        unique_labels = set(dataset_dict['train']['Label']) | set(dataset_dict['test']['Label'])
        label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
        print(f"Dataset loaded. Found {len(label_to_id)} unique labels.")
        return dataset_dict, label_to_id
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None, None

# Updated function signature and loading mechanism
def get_model_and_tokenizer(model_id: str) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """Loads ProGen2 model using AutoModelForCausalLM and tokenizer."""
    print(f"Loading tokenizer for: {model_id}")
    try:
        # Use AutoTokenizer, trust_remote_code is often needed for custom tokenizers/models
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Add padding token if missing (common for generative models)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                print("Tokenizer missing padding token, setting to EOS token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a generic pad token if EOS is also missing (less common)
                print("Tokenizer missing padding and EOS token, adding a new pad token '<PAD>'.")
                tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                # Important: Need to resize model embeddings later if adding tokens *after* loading model

        print(f"Loading model using AutoModelForCausalLM: {model_id}")
        # Use AutoModelForCausalLM instead of AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            # Optional: Low CPU memory usage for very large models, requires accelerate
            # low_cpu_mem_usage=True
        ).to(device)

        # Resize embeddings if a new token was added to tokenizer *before* model loading
        # (Usually safer to ensure tokenizer is settled before loading model)
        # model.resize_token_embeddings(len(tokenizer)) # Uncomment if you added tokens *after* loading

        # Set model's pad token ID based on tokenizer's pad token ID
        if tokenizer.pad_token_id is not None:
             model.config.pad_token_id = tokenizer.pad_token_id
        else:
             print("Warning: Tokenizer does not have a pad_token_id set.")


        print(f"Model loaded on device: {model.device}")
        # Check actual model dtype after loading with "auto"
        try:
             print(f"Actual model dtype: {next(model.parameters()).dtype}")
        except StopIteration:
             print("Warning: Could not retrieve model dtype, model might be empty.")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model or tokenizer for {model_id}: {e}")
        return None, None

# Updated function signature and hidden state extraction
def get_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sequences: List[str]) -> Optional[torch.Tensor]:
    """
    Extracts embeddings using mean pooling of the last hidden state for ProGen2.
    Takes attention mask into account for correct averaging. Uses AutoModelForCausalLM.
    """
    try:
        # Tokenize sequences
        batch_encoding = tokenizer(
            sequences,
            truncation=True,
            padding=True, # Pad sequences to the longest in the batch
            return_tensors="pt",
            #max_length=max_length
        )
        input_ids = batch_encoding.input_ids.to(device)
        attention_mask = batch_encoding.attention_mask.to(device)

        if input_ids.nelement() == 0: # Check for empty tensor after tokenization
             print("Warning: Input IDs tensor is empty after tokenization. Skipping batch.")
             return None

        with torch.no_grad():
            # Get model outputs, explicitly requesting hidden states
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True  # Request hidden states
            )

        # Extract last hidden state from the 'hidden_states' tuple
        # The tuple contains hidden states for all layers; [-1] gets the last one.
        if not hasattr(model_output, 'hidden_states') or not model_output.hidden_states:
            print("Error: 'hidden_states' not found or empty in model output. Ensure output_hidden_states=True was passed and model supports it.")
            return None

        last_hidden_state = model_output.hidden_states[-1]

        if not isinstance(last_hidden_state, torch.Tensor):
             print(f"Error: Expected hidden state to be a Tensor, but got {type(last_hidden_state)}")
             return None
        if last_hidden_state.nelement() == 0:
             print("Error: Last hidden state tensor is empty.")
             return None

        # --- Mean Pooling with Attention Mask ---
        # Expand attention mask to match hidden state dimensions: [batch_size, sequence_length, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Sum embeddings across sequence length, applying mask (zeros out padding)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        # Sum mask elements to get actual sequence lengths (avoids division by zero for empty sequences)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        # Calculate mean pooled embeddings
        mean_pooled_embeddings = sum_embeddings / sum_mask
        # --- End Mean Pooling ---

        return mean_pooled_embeddings

    except Exception as e:
        print(f"Error during embedding extraction: {e}")
        # Log traceback for debugging if needed
        # import traceback
        # traceback.print_exc()
        return None


# Updated function signature
def process_splits(dataset_dict: datasets.DatasetDict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, label_to_id: Dict[str, int]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Processes train and test splits to get embeddings and labels."""
    results = {}
    model.eval() # Set model to evaluation mode

    for split_name in ['train', 'test']:
        print(f"\nProcessing split: {split_name}...")
        current_dataset_split = dataset_dict[split_name]
        all_embeddings = []
        all_labels_indices = []
        num_examples = len(current_dataset_split)
        processed_count = 0

        for i in range(0, num_examples, batch_size):
            batch_data = current_dataset_split[i:min(i + batch_size, num_examples)]
            batch_sequences = batch_data['Sequence']
            batch_string_labels = batch_data['Label']

            # Ensure sequences and labels are lists, even for the last batch if size 1
            if not isinstance(batch_sequences, list):
                 batch_sequences = [batch_sequences]
            if not isinstance(batch_string_labels, list):
                 batch_string_labels = [batch_string_labels]

            # Filter out None or empty sequences *before* processing
            valid_indices = [idx for idx, seq in enumerate(batch_sequences) if isinstance(seq, str) and seq]
            if not valid_indices:
                print(f"Skipping empty or invalid batch at index {i}")
                continue

            # Select only valid data for this batch
            batch_sequences_valid = [batch_sequences[idx] for idx in valid_indices]
            batch_string_labels_valid = [batch_string_labels[idx] for idx in valid_indices]

            embeddings = get_embeddings(model, tokenizer, batch_sequences_valid)

            if embeddings is not None:
                # Ensure label exists in map before converting
                current_batch_labels = []
                valid_embedding_indices = [] # Keep track of which embeddings correspond to valid labels
                for idx, lbl in enumerate(batch_string_labels_valid):
                    if lbl in label_to_id:
                        current_batch_labels.append(label_to_id[lbl])
                        valid_embedding_indices.append(idx)
                    else:
                        print(f"Warning: Label '{lbl}' not found in label_to_id map. Skipping this sequence.")

                if valid_embedding_indices: # Only proceed if at least one label was valid
                    # Select only the embeddings corresponding to valid labels
                    valid_embeddings = embeddings[valid_embedding_indices]
                    all_embeddings.append(valid_embeddings.cpu())
                    all_labels_indices.extend(current_batch_labels)
                    processed_count += len(valid_embedding_indices)
                else:
                    print(f"Warning: No valid labels found in batch starting at index {i}. Skipping.")

            else:
                print(f"Warning: Failed to get embeddings for batch starting at index {i}. Skipping.")

            # Progress update
            if (i // batch_size + 1) % 50 == 0 or (i + len(batch_sequences_valid)) >= num_examples : # Update every 50 batches or on last batch
                 print(f"  Processed {processed_count} / {num_examples} examples for {split_name} split...")

            # Optional: Clear CUDA cache periodically if memory is extremely tight
            # if device.type == 'cuda' and (i // batch_size + 1) % 100 == 0:
            #     torch.cuda.empty_cache()

        print(f"Finished processing {split_name} split. Collected {len(all_embeddings)} embedding tensors and {len(all_labels_indices)} labels.")

        if all_embeddings and all_labels_indices:
            try:
                # Concatenate collected tensors
                embeddings_tensor = torch.cat(all_embeddings, dim=0)
                labels_tensor = torch.tensor(all_labels_indices, dtype=torch.long)
                print(f"Split '{split_name}' - Final Embeddings shape: {embeddings_tensor.shape}, Final Labels shape: {labels_tensor.shape}")

                # Final sanity check on shapes
                if embeddings_tensor.shape[0] == labels_tensor.shape[0]:
                    results[split_name] = (embeddings_tensor, labels_tensor)
                    print(f"Successfully stored results for {split_name} split.")
                else:
                    print(f"Error: Final mismatch in {split_name} embedding count ({embeddings_tensor.shape[0]}) and label count ({labels_tensor.shape[0]}). Skipping split.")
            except Exception as e:
                 print(f"Error concatenating or converting results for split {split_name}: {e}")

        else:
             print(f"Warning: No valid results generated for {split_name} split.")
             if not all_embeddings: print("  Reason: No embeddings were collected/valid.")
             if not all_labels_indices: print("  Reason: No valid labels were collected.")

    return results

def main():
    """Main execution flow."""
    print("--- Starting Embedding Generation ---")
    # Load data
    dataset_dict, label_to_id = load_data_and_map_labels(dataset_name)
    if dataset_dict is None or label_to_id is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id)
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        sys.exit(1)

    # Process splits
    split_results = process_splits(dataset_dict, model, tokenizer, label_to_id)

    # Free up memory explicitly
    print("\nDeleting model and tokenizer to free memory...")
    del model
    del tokenizer
    gc.collect() # Run garbage collector
    if device.type == 'cuda':
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

    # Save results
    if 'train' in split_results and 'test' in split_results:
        # Ensure results are not empty tuples
        if split_results['train'] and split_results['test']:
            final_data_to_save = {
                # Convert embeddings to float32 numpy arrays for broader compatibility
                'train_embeddings': split_results['train'][0].to(torch.float32).numpy(),
                'train_labels': split_results['train'][1].numpy(),
                'test_embeddings': split_results['test'][0].to(torch.float32).numpy(),
                'test_labels': split_results['test'][1].numpy(),
                'label_to_id': label_to_id # Save the label mapping as well
            }

            print(f"\nSaving final data to {output_pickle_file}...")
            try:
                with open(output_pickle_file, 'wb') as f:
                    pickle.dump(final_data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol for efficiency
                print(f"Successfully saved embeddings and labels to {output_pickle_file}")
                # Verify shapes one last time
                print(f"  Train embeddings shape: {final_data_to_save['train_embeddings'].shape}")
                print(f"  Train labels shape: {final_data_to_save['train_labels'].shape}")
                print(f"  Test embeddings shape: {final_data_to_save['test_embeddings'].shape}")
                print(f"  Test labels shape: {final_data_to_save['test_labels'].shape}")

            except Exception as e:
                print(f"Error saving data to pickle file: {e}")
        else:
             print("Processing yielded empty results for train or test split. Nothing saved.")
    else:
        print("Processing did not yield results for both train and test splits. Nothing saved.")
        if 'train' not in split_results: print("  Reason: 'train' key missing from results.")
        if 'test' not in split_results: print("  Reason: 'test' key missing from results.")

    print("--- Embedding Generation Complete ---")

if __name__ == "__main__":
    main()