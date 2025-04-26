import pickle
import numpy as np
from typing import Optional, Dict # Import Optional and Dict for type hinting
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys

# --- Configuration ---
# Input pickle file created by the previous script
#input_pickle_file = "gLM2_650M.pkl"
input_pickle_file = "progen2-xlarge_lt.pkl"

# Models to evaluate
# Using default parameters for simplicity, consider hyperparameter tuning for better results
# Added max_iter to LogisticRegression and MLPClassifier to avoid convergence warnings
# Reduced verbosity for MLPClassifier
models_to_evaluate = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "MLP Classifier": MLPClassifier(max_iter=500, random_state=42, early_stopping=True, verbose=False)
}

def load_data(pickle_file: str) -> Optional[Dict[str, np.ndarray]]:
    """Loads the embeddings and labels from a pickle file."""
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        # Basic validation
        required_keys = ['train_embeddings', 'train_labels', 'test_embeddings', 'test_labels']
        if not all(key in data for key in required_keys):
            print(f"Error: Pickle file '{pickle_file}' is missing required keys.")
            return None
        print(f"Data loaded successfully from {pickle_file}")
        return data
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {pickle_file}")
        return None
    except Exception as e:
        print(f"Error loading data from pickle file: {e}")
        return None

def main():
    """Loads data, trains models, evaluates, and prints scores."""
    print("--- Starting Model Evaluation ---")
    loaded_data = load_data(input_pickle_file)
    if loaded_data is None:
        sys.exit(1)

    X_train = loaded_data['train_embeddings']
    y_train = loaded_data['train_labels']
    X_test = loaded_data['test_embeddings']
    y_test = loaded_data['test_labels']

    print(f"Train data shape: Embeddings {X_train.shape}, Labels {y_train.shape}")
    print(f"Test data shape: Embeddings {X_test.shape}, Labels {y_test.shape}")

    # Filter out convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for name, model in models_to_evaluate.items():
        print(f"\n--- Training and Evaluating: {name} ---")
        try:
            # Train the model
            print("Training...")
            model.fit(X_train, y_train)

            # Make predictions
            print("Predicting...")
            y_pred = model.predict(X_test)

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            # Use macro F1-score for multi-class classification without favoring larger classes
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            print(f"Results for {name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Macro F1-Score: {f1:.4f}")

            # Optional: Print a more detailed classification report
            # print("\nClassification Report:")
            # print(classification_report(y_test, y_pred, zero_division=0))

        except Exception as e:
            print(f"An error occurred while processing model {name}: {e}")

    # Restore default warning behavior
    warnings.filterwarnings("default", category=ConvergenceWarning)
    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    main()
