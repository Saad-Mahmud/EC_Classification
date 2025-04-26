import datasets
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm
import sys
import itertools # Needed for generating k-mers
from collections import Counter # Can be useful for counting
import os
# --- Configuration ---
dataset_name = "tattabio/ec_classification"
# Output filename for this script's results
output_pickle_file = "ec_classification_kmer_freq_123.pkl" # New filename for k-mer features

# Define the standard protein alphabet (20 amino acids)
# We will only count k-mers made of these standard residues.
STANDARD_PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
print(f"Using standard alphabet for K-mers: '{STANDARD_PROTEIN_ALPHABET}'")
alphabet_size = len(STANDARD_PROTEIN_ALPHABET)

# --- Generate all possible K-mers and mapping ---

def generate_kmers(alphabet: str, k: int) -> List[str]:
    """Generates all possible k-mers from the given alphabet."""
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

# Generate the lists of all possible k-mers
all_1mers = generate_kmers(STANDARD_PROTEIN_ALPHABET, 1)
all_2mers = generate_kmers(STANDARD_PROTEIN_ALPHABET, 2)
all_3mers = generate_kmers(STANDARD_PROTEIN_ALPHABET, 3)

# Create mappings from k-mer string to index in the frequency vector
kmer_to_index_1 = {kmer: i for i, kmer in enumerate(all_1mers)}
kmer_to_index_2 = {kmer: i for i, kmer in enumerate(all_2mers)}
kmer_to_index_3 = {kmer: i for i, kmer in enumerate(all_3mers)}

# Calculate dimensions
dim_1mer = len(all_1mers)
dim_2mer = len(all_2mers)
dim_3mer = len(all_3mers)
total_kmer_dimension = dim_1mer + dim_2mer + dim_3mer

print(f"Generated K-mers:")
print(f"  1-mers: {dim_1mer} ({alphabet_size}^1)")
print(f"  2-mers: {dim_2mer} ({alphabet_size}^2)")
print(f"  3-mers: {dim_3mer} ({alphabet_size}^3)")
print(f"Total K-mer embedding dimension: {total_kmer_dimension}")


# --- Combined data loading and label mapping function (Reused from original) ---
def load_data_and_map_labels(dataset_name: str) -> Optional[Tuple[datasets.DatasetDict, Dict[str, int]]]:
    """Loads dataset and creates label mapping."""
    try:
        # Trust Hugging Face cache if available
        dataset_dict = datasets.load_dataset(dataset_name, trust_remote_code=True)
        # Basic check for required splits
        if 'train' not in dataset_dict or 'test' not in dataset_dict:
            print("Error: Dataset loaded but missing 'train' or 'test' split.")
            return None
        # Basic check for Label column
        if 'Label' not in dataset_dict['train'].column_names or 'Label' not in dataset_dict['test'].column_names:
             print("Error: Column 'Label' not found in train or test split.")
             return None
        # Basic check for Sequence column
        if 'Sequence' not in dataset_dict['train'].column_names or 'Sequence' not in dataset_dict['test'].column_names:
             print("Error: Column 'Sequence' not found in train or test split.")
             return None


        unique_labels = set(dataset_dict['train']['Label']) | set(dataset_dict['test']['Label'])
        label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
        print(f"Dataset loaded. Found {len(label_to_id)} unique labels.")
        return dataset_dict, label_to_id
    except Exception as e:
        print(f"An error occurred during data loading or label mapping: {e}")
        return None

# --- K-mer Frequency Calculation ---

def calculate_kmer_frequencies(
    sequence: str,
    alphabet: str,
    kmer_map_1: Dict[str, int],
    kmer_map_2: Dict[str, int],
    kmer_map_3: Dict[str, int]
) -> np.ndarray:
    """
    Calculates the concatenated, normalized 1-mer, 2-mer, and 3-mer frequency
    vector for a single protein sequence.

    Args:
        sequence (str): The protein sequence string.
        alphabet (str): The standard amino acid alphabet to consider.
        kmer_map_1 (Dict): Mapping from 1-mer string to index.
        kmer_map_2 (Dict): Mapping from 2-mer string to index.
        kmer_map_3 (Dict): Mapping from 3-mer string to index.

    Returns:
        np.ndarray: A 1D numpy array containing the concatenated normalized
                    k-mer frequencies (float32).
    """
    # 1. Clean sequence: Keep only standard amino acids
    cleaned_sequence = "".join([char for char in sequence.upper() if char in alphabet])
    n = len(cleaned_sequence)

    # Initialize count arrays (using dimensions derived from maps)
    counts_1mer = np.zeros(len(kmer_map_1), dtype=np.int32)
    counts_2mer = np.zeros(len(kmer_map_2), dtype=np.int32)
    counts_3mer = np.zeros(len(kmer_map_3), dtype=np.int32)

    # 2. Count k-mers
    # 1-mers
    for i in range(n):
        kmer = cleaned_sequence[i]
        if kmer in kmer_map_1: # Should always be true after cleaning
             counts_1mer[kmer_map_1[kmer]] += 1

    # 2-mers
    for i in range(n - 1):
        kmer = cleaned_sequence[i:i+2]
        if kmer in kmer_map_2: # Should always be true
             counts_2mer[kmer_map_2[kmer]] += 1

    # 3-mers
    for i in range(n - 2):
        kmer = cleaned_sequence[i:i+3]
        if kmer in kmer_map_3: # Should always be true
             counts_3mer[kmer_map_3[kmer]] += 1

    # 3. Normalize counts to get frequencies
    # Calculate denominators (total number of k-mers found)
    # Avoid division by zero for short sequences
    total_1mers = n
    total_2mers = max(0, n - 1)
    total_3mers = max(0, n - 2)

    freq_1mer = counts_1mer.astype(np.float32) / total_1mers if total_1mers > 0 else np.zeros_like(counts_1mer, dtype=np.float32)
    freq_2mer = counts_2mer.astype(np.float32) / total_2mers if total_2mers > 0 else np.zeros_like(counts_2mer, dtype=np.float32)
    freq_3mer = counts_3mer.astype(np.float32) / total_3mers if total_3mers > 0 else np.zeros_like(counts_3mer, dtype=np.float32)

    # 4. Concatenate frequency vectors
    full_frequency_vector = np.concatenate((freq_1mer, freq_2mer, freq_3mer))

    return full_frequency_vector


# --- Create Dataset with K-mer Embeddings ---

def create_kmer_freq_dataset(
    dataset_dict: datasets.DatasetDict,
    label_to_id: Dict[str, int],
    alphabet: str,
    kmer_map_1: Dict[str, int],
    kmer_map_2: Dict[str, int],
    kmer_map_3: Dict[str, int],
    embedding_dim: int # Pass the calculated total dimension
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generates K-mer frequency embeddings and labels for train/test splits.

    Args:
        dataset_dict: The dataset dictionary.
        label_to_id: Mapping from string labels to integer indices.
        alphabet: The standard amino acid alphabet.
        kmer_map_1: Mapping for 1-mers.
        kmer_map_2: Mapping for 2-mers.
        kmer_map_3: Mapping for 3-mers.
        embedding_dim: The total dimension of the concatenated k-mer vector.


    Returns:
        A dictionary containing train/test splits with k-mer frequency embeddings (2D)
        and labels as NumPy arrays.
    """
    results = {}
    print(f"Target K-mer embedding dimension: {embedding_dim}")

    for split_name in ['train', 'test']:
        if split_name not in dataset_dict:
            print(f"Split '{split_name}' not found. Skipping.")
            continue

        print(f"\n--- Processing split: {split_name} ---")
        current_dataset_split = dataset_dict[split_name]
        num_examples = len(current_dataset_split)
        all_kmer_embeddings = []
        all_labels_indices = []

        for i in tqdm(range(num_examples), desc=f"Calculating K-mer Freq for {split_name}"):
            try:
                sequence = current_dataset_split[i]['Sequence']
                label_str = current_dataset_split[i]['Label']
            except KeyError as e:
                print(f"Error accessing column {e} at index {i} for split '{split_name}'. Skipping example.")
                continue
            except TypeError as e:
                 print(f"Warning: Skipping example {i} in split '{split_name}' due to data issue (likely None): {e}")
                 continue # Skip if sequence or label is None

            if sequence is None or label_str is None:
                print(f"Warning: Skipping example {i} in split '{split_name}' due to None value in Sequence or Label.")
                continue

            # Calculate the k-mer frequency embedding (produces 1D array)
            kmer_embedding = calculate_kmer_frequencies(
                sequence, alphabet, kmer_map_1, kmer_map_2, kmer_map_3
            )
            all_kmer_embeddings.append(kmer_embedding)

            # Map label string to index
            label_index = label_to_id.get(label_str, -1) # Use -1 for unknown labels
            if label_index == -1:
                 print(f"Warning: Unknown label '{label_str}' found at index {i} for split '{split_name}'.")
            all_labels_indices.append(label_index)

        # Stack the results into NumPy arrays for the split
        if all_kmer_embeddings and all_labels_indices:
            try:
                # Stack list of 1D arrays into a 2D array
                embeddings_array_2d = np.stack(all_kmer_embeddings, axis=0)
                labels_array = np.array(all_labels_indices, dtype=np.int64)

                # Validation
                expected_shape = (num_examples, embedding_dim)
                # Adjust expected shape if examples were skipped
                if embeddings_array_2d.shape[0] != num_examples:
                     print(f"Warning: Number of processed examples ({embeddings_array_2d.shape[0]}) differs from total ({num_examples}) for split '{split_name}'. Likely due to skipped examples.")
                     expected_shape = (embeddings_array_2d.shape[0], embedding_dim)


                if embeddings_array_2d.shape == expected_shape and embeddings_array_2d.shape[0] == labels_array.shape[0]:
                    results[split_name] = (embeddings_array_2d, labels_array)
                    print(f"\nCreated K-MER frequency embeddings and labels for {split_name}")
                    print(f"  Embeddings shape: {embeddings_array_2d.shape}") # Should be (num_examples, total_kmer_dimension)
                    print(f"  Labels shape: {labels_array.shape}")
                else:
                     print(f"Error: Mismatch after processing '{split_name}'.")
                     print(f"  Expected embedding shape: {expected_shape}")
                     print(f"  Actual embedding shape:   {embeddings_array_2d.shape}")
                     print(f"  Actual label count:     {labels_array.shape[0]}")

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

    # K-mer lists and maps are already generated globally

    # Generate k-mer frequency embeddings
    kmer_freq_results = create_kmer_freq_dataset(
        dataset_dict,
        label_to_id,
        STANDARD_PROTEIN_ALPHABET,
        kmer_to_index_1,
        kmer_to_index_2,
        kmer_to_index_3,
        total_kmer_dimension # Pass total dimension
    )

    # Prepare and save results
    if 'train' not in kmer_freq_results or 'test' not in kmer_freq_results:
        print("Error: Missing train or test results after processing. Cannot save.")
        # You might want to investigate why a split is missing if it was expected
        if 'train' in dataset_dict or 'test' in dataset_dict:
             print("Check processing logs for errors during k-mer calculation.")
        sys.exit(1)

    # Use the SAME keys as the other embedding scripts for compatibility
    final_data_to_save = {
        'train_embeddings': kmer_freq_results['train'][0], # Store KMER vector under 'embeddings' key
        'train_labels': kmer_freq_results['train'][1],
        'test_embeddings': kmer_freq_results['test'][0],   # Store KMER vector under 'embeddings' key
        'test_labels': kmer_freq_results['test'][1],
    }

    print(f"\n--- Preparing data for saving to {output_pickle_file} ---")
    # ---> ADD THIS SECTION <---
    
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(final_data_to_save, f)
        print(f"Successfully saved K-MER frequency embeddings and labels to {output_pickle_file}")
        # Print shapes being saved for verification
        print(f"  Saved 'train_embeddings' shape: {final_data_to_save['train_embeddings'].shape}") # Should be 2D (num_seq, total_kmer_dim)
        print(f"  Saved 'train_labels' shape: {final_data_to_save['train_labels'].shape}")
        print(f"  Saved 'test_embeddings' shape: {final_data_to_save['test_embeddings'].shape}") # Should be 2D (num_seq, total_kmer_dim)
        print(f"  Saved 'test_labels' shape: {final_data_to_save['test_labels'].shape}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        sys.exit(1)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()