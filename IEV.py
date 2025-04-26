import datasets
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm
import sys

# --- Configuration ---
dataset_name = "tattabio/ec_classification"
# Output filename for this script's results
output_pickle_file = "ec_classification_integer_encoded.pkl" # New filename

# Define the standard protein alphabet plus padding and common ambiguous characters
# Order matters for the mapping. Padding ('-') should ideally map to 0.
PROTEIN_ALPHABET = "-ACDEFGHIKLMNPQRSTVWYBZXOU" # Put padding '-' first to map to 0

# --- Alphabet and Mapping ---
# Map characters to integers (0 for padding, 1 onwards for amino acids)
char_to_int = {char: i for i, char in enumerate(PROTEIN_ALPHABET)}
alphabet_size = len(PROTEIN_ALPHABET) # Still useful to know size
padding_value = char_to_int['-'] # Should be 0 based on alphabet order
print(f"Using alphabet: '{PROTEIN_ALPHABET}'")
print(f"Integer mapping created (Padding='{padding_value}')")


# --- Combined data loading and label mapping function ---
def load_data_and_map_labels(dataset_name: str) -> Optional[Tuple[datasets.DatasetDict, Dict[str, int]]]:
    """Loads dataset and creates label mapping."""
    try:
        dataset_dict = datasets.load_dataset(dataset_name)
        # Basic check for required splits
        if 'train' not in dataset_dict or 'test' not in dataset_dict:
            print("Error: Dataset loaded but missing 'train' or 'test' split.")
            return None
        # Basic check for Label column
        if 'Label' not in dataset_dict['train'].column_names or 'Label' not in dataset_dict['test'].column_names:
             print("Error: Column 'Label' not found in train or test split.")
             return None

        unique_labels = set(dataset_dict['train']['Label']) | set(dataset_dict['test']['Label'])
        label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
        print(f"Dataset loaded. Found {len(label_to_id)} unique labels.")
        return dataset_dict, label_to_id
    except Exception as e:
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
        return 0
    return max_len

def integer_encode_sequence(sequence: str, char_map: Dict[str, int], max_len: int, pad_val: int) -> np.ndarray:
    """
    Integer encodes a single sequence with padding/truncation.

    Args:
        sequence (str): The protein sequence string.
        char_map (Dict[str, int]): Mapping from amino acid character to integer.
        max_len (int): The target length for padding/truncation.
        pad_val (int): The integer value to use for padding.

    Returns:
        np.ndarray: A 1D numpy array of shape (max_len,).
    """
    # Initialize with padding value
    encoded = np.full((max_len,), fill_value=pad_val, dtype=np.int32) # Use int32 or int16
    sequence = sequence.upper()
    unknown_char_val = char_map.get('X', pad_val) # Use padding value if X not in map

    len_seq = len(sequence)
    for i in range(max_len):
        if i < len_seq:
            char = sequence[i]
            # Get integer for char, default to unknown value if not in map
            encoded[i] = char_map.get(char, unknown_char_val)
        # else: remains padding value

    return encoded

def create_integer_encoded_dataset(
    dataset_dict: datasets.DatasetDict,
    label_to_id: Dict[str, int],
    char_map: Dict[str, int],
    max_len: int, # Use dynamically calculated max_len
    pad_val: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generates integer encoded sequences (1D vectors) and labels for train/test splits.

    Args:
        dataset_dict: The dataset dictionary.
        label_to_id: Mapping from string labels to integer indices.
        char_map: Mapping from amino acid character to integer.
        max_len: The target sequence length (max length found in data).
        pad_val: Integer value used for padding.

    Returns:
        A dictionary containing train/test splits with integer encoded sequences (2D)
        and labels as NumPy arrays.
    """
    results = {}
    print(f"Target vector length (max sequence length): {max_len}")

    for split_name in ['train', 'test']:
        if split_name not in dataset_dict:
            print(f"Split '{split_name}' not found. Skipping.")
            continue

        print(f"\n--- Processing split: {split_name} ---")
        current_dataset_split = dataset_dict[split_name]
        num_examples = len(current_dataset_split)
        all_integer_sequences = []
        all_labels_indices = []

        for i in tqdm(range(num_examples), desc=f"Encoding {split_name}"):
            try:
                sequence = current_dataset_split[i]['Sequence']
                label_str = current_dataset_split[i]['Label']
            except KeyError as e:
                print(f"Error accessing column {e} at index {i} for split '{split_name}'. Skipping example.")
                continue

            # Integer encode the sequence (produces 1D array)
            encoded_1d = integer_encode_sequence(sequence, char_map, max_len, pad_val)
            all_integer_sequences.append(encoded_1d)

            # Map label string to index
            label_index = label_to_id.get(label_str, -1) # Use -1 for unknown labels
            if label_index == -1:
                 print(f"Warning: Unknown label '{label_str}' found at index {i} for split '{split_name}'.")
            all_labels_indices.append(label_index)

        # Stack the results into NumPy arrays for the split
        if all_integer_sequences and all_labels_indices:
            try:
                # Stack list of 1D arrays into a 2D array
                sequences_array_2d = np.stack(all_integer_sequences, axis=0)
                labels_array = np.array(all_labels_indices, dtype=np.int64)

                # Validation
                if sequences_array_2d.shape[0] == labels_array.shape[0] and sequences_array_2d.shape[1] == max_len:
                    results[split_name] = (sequences_array_2d, labels_array)
                    print(f"\nCreated INTEGER encoded sequences and labels for {split_name}")
                    print(f"  Sequences shape: {sequences_array_2d.shape}") # Should be (num_examples, max_len)
                    print(f"  Labels shape: {labels_array.shape}")
                else:
                     print(f"Error: Mismatch in sequence ({sequences_array_2d.shape}) or label ({labels_array.shape[0]}) counts/shape for split '{split_name}'. Expected sequence shape[1]={max_len}")

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

    # Generate integer encoded data
    integer_encoded_results = create_integer_encoded_dataset(
        dataset_dict,
        label_to_id,
        char_to_int,
        max_len, # Use calculated max_len
        padding_value # Pass padding value (usually 0)
    )

    # Prepare and save results
    if 'train' not in integer_encoded_results or 'test' not in integer_encoded_results:
        print("Error: Missing train or test results after processing. Cannot save.")
        sys.exit(1)

    # Use the SAME keys as the GLM-2 embedding script for compatibility
    final_data_to_save = {
        'train_embeddings': integer_encoded_results['train'][0], # Store INTEGER sequence under 'embeddings' key
        'train_labels': integer_encoded_results['train'][1],
        'test_embeddings': integer_encoded_results['test'][0],   # Store INTEGER sequence under 'embeddings' key
        'test_labels': integer_encoded_results['test'][1],
    }

    print(f"\n--- Preparing data for saving to {output_pickle_file} ---")
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(final_data_to_save, f)
        print(f"Successfully saved INTEGER encoded data and labels to {output_pickle_file}")
        # Print shapes being saved for verification
        print(f"  Saved 'train_embeddings' shape: {final_data_to_save['train_embeddings'].shape}") # Should be 2D (num_seq, max_len)
        print(f"  Saved 'train_labels' shape: {final_data_to_save['train_labels'].shape}")
        print(f"  Saved 'test_embeddings' shape: {final_data_to_save['test_embeddings'].shape}") # Should be 2D (num_seq, max_len)
        print(f"  Saved 'test_labels' shape: {final_data_to_save['test_labels'].shape}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        sys.exit(1)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()
