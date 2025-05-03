import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np



models_to_evaluate = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "MLP Classifier": MLPClassifier(max_iter=1000, learning_rate_init=1e-3,
                                    alpha=0.01,
                                    hidden_layer_sizes=(512,256,128), 
                                    random_state=42, 
                                    early_stopping=True, 
                                    verbose=False)
}

def load_data(pickle_file: str):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data
    
def analyze(input_pickle_file, outfile):
    loaded_data = load_data(input_pickle_file)

    X_train = loaded_data['train_embeddings']
    y_train = loaded_data['train_labels']
    X_test = loaded_data['test_embeddings']
    y_test = loaded_data['test_labels']

    print(f"Train data shape: Embeddings {X_train.shape}, Labels {y_train.shape}")
    print(f"Test data shape: Embeddings {X_test.shape}, Labels {y_test.shape}")
    results = {}
    for name, model in models_to_evaluate.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc1 = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)               
            top_k = min(5, probs.shape[1])                    
            topk_ix = np.argsort(probs, axis=1)[:,-top_k:]
            hits = [ y_true in model.classes_[topk_ix[i]]for i, y_true in enumerate(y_test)]
            acc5 = np.mean(hits)
        results[name] = (acc1, f1, acc5)

    with open(outfile, 'w') as f:
        f.write("Model Evaluation Results\n")
        for name, (acc1, f1, acc5) in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Top-1 Accuracy: {acc1:.3f}\n")
            f.write(f"  Macro F1-Score: {f1:.3f}\n")
            f.write(f"  Top-5 Accuracy: {acc5:.3f}\n")
            f.write("\n")
    print(f"Results written to {outfile}")
