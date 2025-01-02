import numpy as np
import json
import matplotlib.pyplot as plt
from train_beamsearch import train
import torch
from dataset import SCANDataset
from torch.utils.data import Subset


def get_add_prim_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3."""
    base_path = "/teamspace/studios/this_studio/Transformer_SCAN/data/add_prim_split"

    pairs = [
        (f"{base_path}/tasks_train_addprim_jump.txt", f"{base_path}/tasks_test_addprim_jump.txt", "jump"),
        (f"{base_path}/tasks_train_addprim_turn_left.txt", f"{base_path}/tasks_test_addprim_turn_left.txt", "turn_left"),
    ]

    additional_base_path = f"{base_path}/with_additional_examples"
    num_composed_commands = ["num1", "num2", "num4", "num8", "num16", "num32"]
    for num in num_composed_commands:
        train_path = f"{additional_base_path}/tasks_train_addprim_complex_jump_{num}.txt"
        test_path = f"{additional_base_path}/tasks_test_addprim_complex_jump_{num}.txt"
        pairs.append((train_path, test_path, f"jump_{num}"))

    return pairs


def save_results(results, filename="experiment3_results.json"):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")


def plot_histograms(results, filename_prefix="experiment3"):
    """Plot token and sequence accuracy histograms for Experiment 3."""
    datasets = []
    token_means = []
    token_stds = []
    seq_means = []
    seq_stds = []

    for dataset, accuracies in results.items():
        valid_accuracies = [acc for acc in accuracies if acc is not None]
        if valid_accuracies:
            token_accuracies = [acc[0] for acc in valid_accuracies]
            seq_accuracies = [acc[1] for acc in valid_accuracies]

            datasets.append(dataset)
            token_means.append(np.mean(token_accuracies))
            token_stds.append(np.std(token_accuracies))
            seq_means.append(np.mean(seq_accuracies))
            seq_stds.append(np.std(seq_accuracies))

    # Token Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, token_means, yerr=token_stds, capsize=5, alpha=0.7, label="Token Accuracy", color="blue")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Token Accuracy for Experiment 3")
    plt.legend()
    plt.savefig(f"{filename_prefix}_token_accuracy.png")
    plt.close()

    # Sequence Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, seq_means, yerr=seq_stds, capsize=5, alpha=0.7, label="Sequence Accuracy", color="orange")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Sequence Accuracy for Experiment 3")
    plt.legend()
    plt.savefig(f"{filename_prefix}_sequence_accuracy.png")
    plt.close()


def run_experiment_3(n_runs=1):
    """Run Experiment 3 with beam search and multiple dataset variations."""
    results = {}

    # Hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 8,
        "forward_dim": 256,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Get dataset pairs
    dataset_pairs = get_add_prim_dataset_pairs()

    for train_path, test_path, dataset_name in dataset_pairs:
        print(f"\nTraining on dataset: {dataset_name}")
        results[dataset_name] = []

        for run in range(n_runs):
            seed = 42 + run
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            # Load datasets
            train_dataset = SCANDataset(train_path)
            test_dataset = SCANDataset(test_path)

            # Use a subset of the test dataset
            subset_size = len(test_dataset) // 3  # 33% because of computation power limitations
            test_subset = Subset(test_dataset, range(subset_size))

            # Run training
            try:
                _, token_acc, seq_acc = train(
                    train_path=train_dataset,
                    test_path=test_subset,
                    hyperparams=hyperparams,
                    model_suffix=dataset_name,
                    random_seed=seed,
                )
                results[dataset_name].append((token_acc, seq_acc))
                print(f"Run {run + 1}/{n_runs}: Token Accuracy = {token_acc:.4f}, Sequence Accuracy = {seq_acc:.4f}")

            except Exception as e:
                print(f"Error during training/testing for {dataset_name}: {e}")
                results[dataset_name].append(None)

    # Save results and plot histograms
    save_results(results)
    plot_histograms(results)

    # Print results summary
    print("\nFinal Results Summary:")
    for dataset, accuracies in results.items():
        valid_accuracies = [acc for acc in accuracies if acc is not None]
        if valid_accuracies:
            token_accuracies = [acc[0] for acc in valid_accuracies]
            seq_accuracies = [acc[1] for acc in valid_accuracies]
            print(f"{dataset}:")
            print(f"  Token Accuracy: Mean = {np.mean(token_accuracies):.4f}, Std = {np.std(token_accuracies):.4f}")
            print(f"  Sequence Accuracy: Mean = {np.mean(seq_accuracies):.4f}, Std = {np.std(seq_accuracies):.4f}")
        else:
            print(f"{dataset}: No valid results (errors encountered)")

if __name__ == "__main__":
    run_experiment_3(n_runs=1)
