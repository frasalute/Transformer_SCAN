import torch
import numpy as np
from train_greedysearch import train
import matplotlib.pyplot as plt


def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
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
        "epochs":50,
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
            
            # Call the main function
            try:
                g_accuracy, seq_acc, accuracy = train(
                    train_path=train_path,
                    test_path=test_path,
                    hyperparams=hyperparams,
                    model_suffix=f"p_{size}",
                    random_seed=seed
                )
                results[f"p{size}"].append((accuracy, g_accuracy))
            except TypeError as e:
                print(f"Error during training for size {size}: {e}")
                continue

    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev")
    print("-" * 50)

    for size, accuracies in results.items():
        # Ensure compatibility with tensor and non-tensor values
        accuracies = [(acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                       g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc)
                      for acc, g_acc in accuracies]
        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)
        print(f"{size:11} | Mean Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
        print(f"Mean Greedy Accuracy: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")

       # Add histogram generation at the end
        sizes = [size[1:] for size in results.keys()]  # Extract dataset sizes
        teacher_means = [np.mean([(acc[0] if not torch.is_tensor(acc[0]) else acc[0].cpu().numpy())
                                for acc in results[size]]) for size in results.keys()]
        teacher_stds = [np.std([(acc[0] if not torch.is_tensor(acc[0]) else acc[0].cpu().numpy())
                                for acc in results[size]]) for size in results.keys()]
        greedy_means = [np.mean([(acc[1] if not torch.is_tensor(acc[1]) else acc[1].cpu().numpy())
                                for acc in results[size]]) for size in results.keys()]
        greedy_stds = [np.std([(acc[1] if not torch.is_tensor(acc[1]) else acc[1].cpu().numpy())
                            for acc in results[size]]) for size in results.keys()]

        x = np.arange(len(sizes))  # Positions for the bars
        width = 0.35  # Bar width

        plt.figure(figsize=(12, 8))

        # Teacher bars
        plt.bar(x - width / 2, teacher_means, width, yerr=teacher_stds, capsize=5,
                label="Teacher", color="teal", alpha=0.7)

        # Greedy bars
        plt.bar(x + width / 2, greedy_means, width, yerr=greedy_stds, capsize=5,
                label="Greedy", color="gold", alpha=0.7)

        # Add labels, title, legend
        plt.xlabel("Dataset Size (%)", fontsize=14)
        plt.ylabel("Accuracy (%)", fontsize=14)
        plt.title("Mean Accuracy with Standard Deviation", fontsize=16)
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
