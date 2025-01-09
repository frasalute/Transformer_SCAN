import torch
import numpy as np
from train_greedysearch import train
import matplotlib.pyplot as plt


def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "/teamspace/studios/this_studio/Transformer_SCAN/data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs

def run_all_variations(n_runs=1):
    """Run training multiple times for all dataset size variations with different seeds."""
    results = {}

    # Initialize hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 1,
        "n_heads": 8,
        "forward_dim": 512,
        "dropout": 0.05,
        "learning_rate": 7e-4,
        "batch_size": 64,
        "epochs": 50,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Initialize results dictionary
    for _, _, size in get_dataset_pairs():
        results[f"p{size}"] = []

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run + 1}/{n_runs} with seed {seed}")
        print("=" * 70)

        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            print(f"Train path: {train_path}, Test path: {test_path}")
            
            # Call the train function
            try:
                model, token_acc, seq_acc = train(
                    train_path=train_path,
                    test_path=test_path,
                    hyperparams=hyperparams,
                    model_suffix=f"p_{size}",
                    random_seed=seed
                )
                results[f"p{size}"].append((token_acc, seq_acc))
            except Exception as e:
                print(f"Error during training for size {size}: {e}")
                continue

    # Final Results Summary
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Token Accuracy ± Std Dev | Mean Sequence Accuracy ± Std Dev")
    print("-" * 50)

    # Prepare data for the graph
    sizes = []
    token_means = []
    token_stds = []
    seq_means = []
    seq_stds = []

    for size, accuracies in results.items():
        token_accuracies = [acc[0] for acc in accuracies]
        seq_accuracies = [acc[1] for acc in accuracies]

        token_mean = np.mean(token_accuracies)
        token_std = np.std(token_accuracies)
        seq_mean = np.mean(seq_accuracies)
        seq_std = np.std(seq_accuracies)

        token_means.append(token_mean)
        token_stds.append(token_std)
        seq_means.append(seq_mean)
        seq_stds.append(seq_std)
        sizes.append(size[1:])  # Remove 'p' prefix for graph labels

        print(f"{size:11} | {token_mean:.4f} ± {token_std:.4f} | {seq_mean:.4f} ± {seq_std:.4f}")
        print("-" * 50)

    # Generate the bar graph
    x = np.arange(len(sizes))  # Positions for the bars
    width = 0.35  # Bar width

    plt.figure(figsize=(12, 8))

    # Token bars
    plt.bar(x - width / 2, token_means, width, yerr=token_stds, capsize=5,
            label="Token Accuracy", color="teal", alpha=0.7)

    # Sequence bars
    plt.bar(x + width / 2, seq_means, width, yerr=seq_stds, capsize=5,
            label="Sequence Accuracy", color="gold", alpha=0.7)

    # Add labels, title, legend
    plt.xlabel("Dataset Size (%)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Accuracy for Token and Sequence", fontsize=16)
    plt.xticks(x, labels=[f"{size}%" for size in sizes], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and show plot
    plt.tight_layout()
    plt.savefig("experiment1_results_plot.png")
    plt.show()

if __name__ == "__main__":
    run_all_variations()
