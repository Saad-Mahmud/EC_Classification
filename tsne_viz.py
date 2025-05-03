import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    input_pickle = "data/KM.pkl"
    with open(input_pickle, "rb") as f:
        data = pickle.load(f)
    X = data[f"{"train"}_embeddings"]
    y = data[f"{"train"}_labels"]

    tsne = TSNE(n_components=2, n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=10)
    plt.title(f"t-SNE of {"train"} embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Class label")

    plt.tight_layout()
    
    plt.savefig("tsne.png", dpi=300)
    
if __name__ == "__main__":
    main()

