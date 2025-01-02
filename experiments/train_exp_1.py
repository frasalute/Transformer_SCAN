import sys
import json
import matplotlib.pyplot as plt
sys.path.append('/work/ATNLP-Project')
from train_beamsearch import train
import torch
import numpy as np
from dataset import SCANDataset
from torch.utils.data import DataLoader, Subset


def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "/work/ATNLP-Project/data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs


def save_results(results, filename="results.json"):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")


def plot_histograms(results, filename_prefix="histogram"):
    """Plot histograms for token and sequence accuracy."""
    dataset_sizes = []
    token_means = []
    token_stds = []
    seq_means = []
    seq_stds = []

    for size, accuracies in results.items():
        valid_accuracies = [acc for acc in accuracies if acc is not None]
        if valid_accuracies:
            token_accuracies = [acc[0] for acc in valid_accuracies]
            seq_accuracies = [acc[1] for acc in valid_accuracies]

            dataset_sizes.append(size)
            token_means.append(np.mean(token_accuracies))
            token_stds.append(np.std(token_accuracies))
            seq_means.append(np.mean(seq_accuracies))
            seq_stds.append(np.std(seq_accuracies))

    # Plot Token Accuracy Histogram
    plt.figure(figsize=(10, 6))
    plt.bar(dataset_sizes, token_means, yerr=token_stds, capsize=5, alpha=0.7, label="Token Accuracy")
    plt.xlabel("Dataset Size")
    plt.ylabel("Accuracy")
    plt.title("Token Accuracy by Dataset Size")
    plt.legend()
    plt.savefig(f"{filename_prefix}_token_accuracy.png")
    print(f"Token accuracy histogram saved to {filename_prefix}_token_accuracy.png")
    plt.close()

    # Plot Sequence Accuracy Histogram
    plt.figure(figsize=(10, 6))
    plt.bar(dataset_sizes, seq_means, yerr=seq_stds, capsize=5, alpha=0.7, label="Sequence Accuracy", color="orange")
    plt.xlabel("Dataset Size")
    plt.ylabel("Accuracy")
    plt.title("Sequence Accuracy by Dataset Size")
    plt.legend()
    plt.savefig(f"{filename_prefix}_sequence_accuracy.png")
    print(f"Sequence accuracy histogram saved to {filename_prefix}_sequence_accuracy.png")
    plt.close()


def run_all_variations(n_runs=1):
    """Run training multiple times for all dataset size variations with different seeds."""
    results = {f"p{size}": [] for _, _, size in get_dataset_pairs()}

    # Initialize hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 1,
        "n_heads": 8,
        "forward_dim": 512,
        "dropout": 0.05,
        "learning_rate": 7e-4,
        "batch_size": 64,
         "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run + 1}/{n_runs} with seed {seed}")
        print("=" * 70)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            try:
                # Dynamically calculate epochs
                train_dataset = SCANDataset(train_path)
                test_dataset = SCANDataset(test_path)  
                dataset_size = len(train_dataset)
                epochs = max(1, 100000 // dataset_size)
                hyperparams["epochs"] = epochs

                print(f"Dataset size: {dataset_size}, Training for {epochs} epochs")

                # Subset the test data for beam search
                subset_size = len(test_dataset) // 3 # 33% because I don't have a lot of computational power
                test_subset = Subset(test_dataset, range(subset_size))
                test_loader = DataLoader(test_subset, batch_size=hyperparams["batch_size"], shuffle=False)

                # Call train with all required arguments
                model_suffix = f"p_{size}"
                model, token_acc, seq_acc = train(
                    train_path=train_dataset,
                    test_path=test_loader,  # Pass test_loader for evaluation
                    hyperparams=hyperparams,
                    model_suffix=model_suffix,
                    random_seed=seed,
                )

                # Store results
                results[f"p{size}"].append((float(token_acc), float(seq_acc)))

            except Exception as e:
                print(f"Error during training/testing for p{size}: {e}")
                results[f"p{size}"].append(None)

    # Save results to file
    save_results(results)

    # Print results summary
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Token Accuracy ± Std Dev | Mean Sequence Accuracy ± Std Dev")
    print("-" * 50)

    for size, accuracies in results.items():
        valid_accuracies = [acc for acc in accuracies if acc is not None]
        if valid_accuracies:
            token_accuracies = [acc[0] for acc in valid_accuracies]
            seq_accuracies = [acc[1] for acc in valid_accuracies]

            mean_token_acc = np.mean(token_accuracies)
            std_token_acc = np.std(token_accuracies)
            mean_seq_acc = np.mean(seq_accuracies)
            std_seq_acc = np.std(seq_accuracies)

            print(f"{size:11} | {mean_token_acc:.4f} ± {std_token_acc:.4f}          | {mean_seq_acc:.4f} ± {std_seq_acc:.4f}")
        else:
            print(f"{size:11} | No results available (errors encountered)")

    # Generate histograms
    plot_histograms(results)


if __name__ == "__main__":
    run_all_variations(n_runs=1)
