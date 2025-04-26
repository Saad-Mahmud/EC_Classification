import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Tuple # Import Tuple for type hinting
import sys
import time # To time operations

# --- Configuration ---
# Input pickle file (from integer encoding script)
input_pickle_file = "kmer.pkl"
# Output pickle file for PCA-reduced data
output_pickle_file = "kmer.pkl"

# PCA configuration: Retain components explaining 95% of the variance
# Alternatively, set to an integer like n_components=128 for a fixed number
pca_n_components = 0.95

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
        # Print shapes for confirmation
        print(f"  Train embeddings shape: {data['train_embeddings'].shape}")
        print(f"  Test embeddings shape: {data['test_embeddings'].shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {pickle_file}")
        return None
    except Exception as e:
        print(f"Error loading data from pickle file: {e}")
        return None

def apply_pca(X_train: np.ndarray, X_test: np.ndarray, n_components) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies scaling and PCA to the data.

    Args:
        X_train: Training features.
        X_test: Testing features.
        n_components: Parameter for PCA (float for variance or int for number).

    Returns:
        A tuple containing (X_train_pca, X_test_pca, fitted_pca_object).
    """
    print("\n--- Applying StandardScaler ---")
    start_time = time.time()
    scaler = StandardScaler()
    # Fit scaler ONLY on training data
    print("Fitting scaler on training data...")
    scaler.fit(X_train)
    # Transform both train and test data
    print("Transforming data with scaler...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaling completed in {time.time() - start_time:.2f} seconds.")

    print("\n--- Applying PCA ---")
    start_time = time.time()
    # Instantiate PCA
    pca = PCA(n_components=n_components, random_state=42)
    # Fit PCA ONLY on scaled training data
    print(f"Fitting PCA on scaled training data (n_components={n_components})...")
    pca.fit(X_train_scaled)
    # Transform both scaled train and test data
    print("Transforming data with PCA...")
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA completed in {time.time() - start_time:.2f} seconds.")

    print(f"\nPCA Results:")
    print(f"  Number of components selected: {pca.n_components_}")
    if isinstance(n_components, float) and n_components < 1.0:
        print(f"  Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"  Shape after PCA (train): {X_train_pca.shape}")
    print(f"  Shape after PCA (test): {X_test_pca.shape}")

    return X_train_pca, X_test_pca, pca


def main():
    """Loads data, applies PCA, and saves the reduced data."""
    print("--- Starting PCA Reduction Script ---")
    loaded_data = load_data(input_pickle_file)
    if loaded_data is None:
        sys.exit(1)

    X_train = loaded_data['train_embeddings']
    y_train = loaded_data['train_labels']
    X_test = loaded_data['test_embeddings']
    y_test = loaded_data['test_labels']

    # Apply Scaling and PCA
    try:
        X_train_pca, X_test_pca, fitted_pca = apply_pca(X_train, X_test, pca_n_components)
    except Exception as e:
         print(f"\nError during PCA processing: {e}")
         sys.exit(1)


    # Prepare data for saving using the original keys for compatibility
    final_data_to_save = {
        'train_embeddings': X_train_pca, # Store PCA-reduced data
        'train_labels': y_train,        # Keep original labels
        'test_embeddings': X_test_pca,  # Store PCA-reduced data
        'test_labels': y_test,          # Keep original labels
    }

    # Save the PCA-reduced data
    print(f"\n--- Preparing data for saving to {output_pickle_file} ---")
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(final_data_to_save, f)
        print(f"Successfully saved PCA reduced data and labels to {output_pickle_file}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        sys.exit(1)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()
