import datasets
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional, Set # Keep Optional/Tuple for other functions if needed
from tqdm import tqdm
import sys

# --- Configuration ---
dataset_name = "tattabio/ec_classification"
# Output filename for this script's results
output_pickle_file = "ec_classification_one_hot_flattened.pkl" # Updated filename

# Define the standard protein alphabet plus padding and common ambiguous characters
# Order matters for the mapping. '-' (padding) is often last.
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBZXOU-" # B=Asx, Z=Glx, X=Unknown, O=Pyl, U=Sec, -=Padding

# --- Alphabet and Mapping ---
char_to_int = {char: i for i, char in enumerate(PROTEIN_ALPHABET)}
alphabet_size = len(PROTEIN_ALPHABET)
print(f"Using alphabet: '{PROTEIN_ALPHABET}' (Size: {alphabet_size})")

# --- Combined data loading and label mapping function ---
def load_data_and_map_labels(dataset_name: str) -> Optional[Tuple[datasets.DatasetDict, Dict[str, int]]]:
    """Loads dataset and creates label mapping."""
    try:
        dataset_dict = datasets.load_dataset(dataset_name)
        # Basic check for required splits
        if 'train' not in dataset_dict or 'test' not in dataset_dict:
            print("Error: Dataset loaded but missing 'train' or 'test' split.")
            return None # Return None if splits are missing
        # Basic check for Label column
        if 'Label' not in dataset_dict['train'].column_names or 'Label' not in dataset_dict['test'].column_names:
             print("Error: Column 'Label' not found in train or test split.")
             return None # Return None if Label column missing

        unique_labels = set(dataset_dict['train']['Label']) | set(dataset_dict['test']['Label'])
        label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
        print(f"Dataset loaded. Found {len(label_to_id)} unique labels.")
        return dataset_dict, label_to_id
    except Exception as e:
        # Catch potential errors during loading or processing
        print(f"An error occurred during data loading or label mapping: {e}")
        return None

def get_max_sequence_length(dataset_dict: datasets.DatasetDict) -> int:
    """Calculates the maximum sequence length across train and test splits."""
    max_len = 0
    try:
        for split in ['train', 'test']:
            if split in dataset_dict:
                 lengths = [len(seq) for seq in dataset_dict[split]['Sequence']]
                 split_max = max(lengths) if lengths else 0
                 max_len = max(max_len, split_max)
    except KeyError:
        print("Error: Could not access 'Sequence' column to determine max length.")
        # Return a default or raise an error? Let's return 0 and handle in main
        return 0
    return max_len

def one_hot_encode(sequence: str, char_map: Dict[str, int], max_len: int, alphabet_len: int) -> np.ndarray:
    """
    One-hot encodes a single sequence with padding/truncation.

    Args:
        sequence (str): The protein sequence string.
        char_map (Dict[str, int]): Mapping from amino acid character to index.
        max_len (int): The target length for padding/truncation.
        alphabet_len (int): The size of the alphabet (number of columns).

    Returns:
        np.ndarray: A 2D numpy array of shape (max_len, alphabet_len).
    """
    encoded = np.zeros((max_len, alphabet_len), dtype=np.int8)
    sequence = sequence.upper()
    padding_char = '-'
    padding_idx = char_map.get(padding_char)

    for i in range(max_len):
        if i < len(sequence):
            char = sequence[i]
            char_index = char_map.get(char)
            if char_index is not None:
                encoded[i, char_index] = 1
        # else: Padding or unknown chars remain zeros

    return encoded

def create_one_hot_dataset(
    dataset_dict: datasets.DatasetDict,
    label_to_id: Dict[str, int],
    char_map: Dict[str, int],
    max_len: int, # Use dynamically calculated max_len
    alphabet_len: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generates FLATTENED one-hot encoded sequences and labels for train/test splits.

    Args:
        dataset_dict: The dataset dictionary.
        label_to_id: Mapping from string labels to integer indices.
        char_map: Mapping from amino acid character to index.
        max_len: The target sequence length (max length found in data).
        alphabet_len: The size of the alphabet.

    Returns:
        A dictionary containing train/test splits with FLATTENED one-hot sequences (2D)
        and labels as NumPy arrays.
    """
    results = {}
    flattened_vector_size = max_len * alphabet_len
    print(f"Flattened vector size: {max_len} * {alphabet_len} = {flattened_vector_size}")

    for split_name in ['train', 'test']:
        if split_name not in dataset_dict:
            print(f"Split '{split_name}' not found. Skipping.")
            continue

        print(f"\n--- Processing split: {split_name} ---")
        current_dataset_split = dataset_dict[split_name]
        num_examples = len(current_dataset_split)
        # Pre-allocate flattened array if memory allows, otherwise build list
        # Using lists and stacking/reshaping is more flexible
        all_flattened_sequences = []
        all_labels_indices = []

        for i in tqdm(range(num_examples), desc=f"Encoding {split_name}"):
            try:
                sequence = current_dataset_split[i]['Sequence']
                label_str = current_dataset_split[i]['Label']
            except KeyError as e:
                print(f"Error accessing column {e} at index {i} for split '{split_name}'. Skipping example.")
                continue

            # One-hot encode the sequence (produces 2D array)
            encoded_2d = one_hot_encode(sequence, char_map, max_len, alphabet_len)
            # Flatten the 2D array to 1D and append
            all_flattened_sequences.append(encoded_2d.flatten())

            # Map label string to index
            label_index = label_to_id.get(label_str, -1)
            if label_index == -1:
                 print(f"Warning: Unknown label '{label_str}' found at index {i} for split '{split_name}'.")
            all_labels_indices.append(label_index)

        # Stack the results into NumPy arrays for the split
        if all_flattened_sequences and all_labels_indices:
            try:
                # Stack list of 1D arrays into a 2D array
                sequences_array_2d = np.stack(all_flattened_sequences, axis=0)
                labels_array = np.array(all_labels_indices, dtype=np.int64)

                # Validation
                if sequences_array_2d.shape[0] == labels_array.shape[0] and sequences_array_2d.shape[1] == flattened_vector_size:
                    results[split_name] = (sequences_array_2d, labels_array)
                    print(f"\nCreated FLATTENED one-hot sequences and labels for {split_name}")
                    print(f"  Sequences shape: {sequences_array_2d.shape}") # Should be (num_examples, max_len * alphabet_size)
                    print(f"  Labels shape: {labels_array.shape}")
                else:
                     print(f"Error: Mismatch in sequence ({sequences_array_2d.shape}) or label ({labels_array.shape[0]}) counts/shape for split '{split_name}'. Expected sequence shape[1]={flattened_vector_size}")

            except Exception as e:
                print(f"Error stacking results for split {split_name}: {e}")
        else:
            print(f"No sequences or labels processed for split {split_name}.")

    return results


def main():
    """Main execution flow."""
    load_result = load_data_and_map_labels(dataset_name)
    if load_result is None:
        print("Failed to load data or map labels. Exiting.")
        sys.exit(1)
    dataset_dict, label_to_id = load_result

    # Dynamically determine max sequence length
    max_len = get_max_sequence_length(dataset_dict)
    if max_len == 0:
         print("Error: Could not determine maximum sequence length. Exiting.")
         sys.exit(1)
    print(f"Determined Max sequence length: {max_len}")


    # Generate flattened one-hot encoded data
    one_hot_results = create_one_hot_dataset(
        dataset_dict,
        label_to_id,
        char_to_int,
        max_len, # Use calculated max_len
        alphabet_size
    )

    # Prepare and save results
    if 'train' not in one_hot_results or 'test' not in one_hot_results:
        print("Error: Missing train or test results after processing. Cannot save.")
        sys.exit(1)

    # Use the SAME keys as the GLM-2 embedding script for compatibility
    final_data_to_save = {
        'train_embeddings': one_hot_results['train'][0], # Store FLATTENED one-hot under 'embeddings' key
        'train_labels': one_hot_results['train'][1],
        'test_embeddings': one_hot_results['test'][0],   # Store FLATTENED one-hot under 'embeddings' key
        'test_labels': one_hot_results['test'][1],
    }

    print(f"\n--- Preparing data for saving to {output_pickle_file} ---")
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(final_data_to_save, f)
        print(f"Successfully saved FLATTENED one-hot encoded data and labels to {output_pickle_file}")
        # Print shapes being saved for verification
        print(f"  Saved 'train_embeddings' shape: {final_data_to_save['train_embeddings'].shape}") # Should be 2D
        print(f"  Saved 'train_labels' shape: {final_data_to_save['train_labels'].shape}")
        print(f"  Saved 'test_embeddings' shape: {final_data_to_save['test_embeddings'].shape}") # Should be 2D
        print(f"  Saved 'test_labels' shape: {final_data_to_save['test_labels'].shape}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        sys.exit(1)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()
