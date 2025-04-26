import pickle
import numpy as np
# Matplotlib, Seaborn, Pandas are no longer needed
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Tuple # Import Tuple
import sys
import time # To time operations

# --- Configuration ---
# Input pickle file (from integer encoding script)
input_pickle_file = "kmer.pkl"
# Output pickle file for t-SNE reduced data
output_pickle_file = "kmer_tsne.pkl" # New output filename

# t-SNE Parameters (adjust as needed)
tsne_n_components = 128      # Reduce to 2 dimensions (can be changed)
tsne_perplexity = 30.0   # Related to number of nearest neighbors, typical range 5-50
tsne_learning_rate = 'auto' # Often works well, otherwise try values like 200
tsne_n_iter = 1000     # Number of optimization iterations
tsne_init = 'pca'      # PCA initialization can be more stable than 'random'
tsne_method = 'exact'  # Explicitly set method to 'exact' to bypass barnes_hut check
tsne_random_state = 42 # For reproducibility

print(f"Using t-SNE method: '{tsne_method}'") # Notify user which method is used

def load_data(pickle_file: str) -> Optional[Dict[str, np.ndarray]]:
    """Loads the encoded sequences and labels from a pickle file."""
    try:
        print(f"Loading data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        # Basic validation
        required_keys = ['train_embeddings', 'train_labels', 'test_embeddings', 'test_labels']
        if not all(key in data for key in required_keys):
            print(f"Error: Pickle file '{pickle_file}' is missing required keys.")
            return None
        print(f"Data loaded successfully.")
        print(f"  Train embeddings shape: {data['train_embeddings'].shape}")
        print(f"  Test embeddings shape: {data['test_embeddings'].shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {pickle_file}")
        return None
    except Exception as e:
        print(f"Error loading data from pickle file: {e}")
        return None

def apply_tsne(X_train: np.ndarray, X_test: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Applies scaling and t-SNE separately to train and test sets.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        A tuple containing (X_train_tsne, X_test_tsne), or None on error.
    """
    try:
        print("\n--- Applying StandardScaler ---")
        start_time = time.time()
        scaler = StandardScaler()
        # Fit scaler ONLY on training data, transform both
        print("Scaling training and test data...")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Use transform for test set
        print(f"Scaling completed in {time.time() - start_time:.2f} seconds.")

        print("\n--- Applying t-SNE ---")
        print(f"Parameters: n_components={tsne_n_components}, perplexity={tsne_perplexity}, n_iter={tsne_n_iter}, init='{tsne_init}', learning_rate='{tsne_learning_rate}', method='{tsne_method}'")

        # Instantiate t-SNE, explicitly setting method='exact'
        tsne = TSNE(n_components=tsne_n_components,
                    perplexity=tsne_perplexity,
                    learning_rate=tsne_learning_rate,
                    n_iter=tsne_n_iter,
                    init=tsne_init,
                    method=tsne_method, # Use the configured method
                    random_state=tsne_random_state,
                    n_jobs=-1) # Use all available CPU cores

        # Apply fit_transform to Training Data
        print("Applying t-SNE to training data (fit_transform)...")
        start_time = time.time()
        X_train_tsne = tsne.fit_transform(X_train_scaled)
        print(f"t-SNE on training data completed in {time.time() - start_time:.2f} seconds.")
        print(f"Shape after t-SNE (train): {X_train_tsne.shape}")

        # Apply fit_transform SEPARATELY to Test Data
        # ** WARNING: Test embedding is NOT comparable to train embedding **
        # ** t-SNE does not learn a reusable transformation function **
        print("\nApplying t-SNE to test data (fit_transform)...")
        print("WARNING: Test set t-SNE is calculated independently and is not directly comparable to the training set embedding space.")
        start_time = time.time()
        # Reusing the same tsne object configuration but applying fit_transform again
        X_test_tsne = tsne.fit_transform(X_test_scaled) # Applying fit_transform again
        print(f"t-SNE on test data completed in {time.time() - start_time:.2f} seconds.")
        print(f"Shape after t-SNE (test): {X_test_tsne.shape}")

        return X_train_tsne, X_test_tsne

    except Exception as e:
        print(f"Error during t-SNE processing: {e}")
        return None


def main():
    """Loads data, applies t-SNE, and saves the reduced data."""
    print("--- Starting t-SNE Embedding Generation Script ---")
    loaded_data = load_data(input_pickle_file)
    if loaded_data is None:
        sys.exit(1)

    X_train = loaded_data['train_embeddings']
    y_train = loaded_data['train_labels']
    X_test = loaded_data['test_embeddings']
    y_test = loaded_data['test_labels']

    # Apply Scaling and t-SNE
    tsne_results = apply_tsne(X_train, X_test)
    if tsne_results is None:
         print("\nError during t-SNE processing. Exiting.")
         sys.exit(1)
    X_train_tsne, X_test_tsne = tsne_results


    # Prepare data for saving using the standard keys for compatibility
    final_data_to_save = {
        'train_embeddings': X_train_tsne, # Store t-SNE reduced data
        'train_labels': y_train,         # Keep original labels
        'test_embeddings': X_test_tsne,  # Store t-SNE reduced data (use with caution!)
        'test_labels': y_test,           # Keep original labels
    }

    # Save the t-SNE reduced data
    print(f"\n--- Preparing data for saving to {output_pickle_file} ---")
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(final_data_to_save, f)
        print(f"Successfully saved t-SNE reduced data and labels to {output_pickle_file}")
        print(f"  Saved 'train_embeddings' shape: {final_data_to_save['train_embeddings'].shape}")
        print(f"  Saved 'test_embeddings' shape: {final_data_to_save['test_embeddings'].shape}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        sys.exit(1)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()
