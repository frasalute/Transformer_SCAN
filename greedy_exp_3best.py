import numpy as np
import torch
from train_greedysearch import train, greedy_decode, calculate_accuracy
from dataset import SCANDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print
import matplotlib.pyplot as plt

def get_add_prim_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3."""
    base_path = "/work/Transformer_SCAN/data/add_prim_split"

    # Basic datasets
    pairs = [
        (
            [(f"{base_path}/tasks_train_addprim_jump.txt", f"{base_path}/tasks_test_addprim_jump.txt")],
            "jump",
        ),
        (
            [(f"{base_path}/tasks_train_addprim_turn_left.txt", f"{base_path}/tasks_test_addprim_turn_left.txt")],
            "turn_left",
        ),
    ]

    # Additional datasets with repetitions
    additional_base_path = "/work/Transformer_SCAN/data/add_prim_split/with_additional_examples"
    num_composed_commands = ["num1", "num2", "num4", "num8", "num16", "num32"]
    for num in num_composed_commands:
        train_test_pairs = []
        for rep in range(1, 6):  # 5 repetitions
            train_path = f"{additional_base_path}/tasks_train_addprim_complex_jump_{num}_rep{rep}.txt"
            test_path = f"{additional_base_path}/tasks_test_addprim_complex_jump_{num}_rep{rep}.txt"
            train_test_pairs.append((train_path, test_path))
        pairs.append((train_test_pairs, num))

    return pairs

def evaluate(model, test_loader, hyperparams):
    """Evaluate the model using greedy search."""
    pad_idx = test_loader.dataset.vocab.special_tokens["<PAD>"]
    bos_idx = test_loader.dataset.vocab.special_tokens["<BOS>"]
    eos_idx = test_loader.dataset.vocab.special_tokens["<EOS>"]

    token_accs = []
    seq_accs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            # Greedy decoding
            pred = greedy_decode(
                model, src, max_len=tgt.size(1), start_symbol=bos_idx, end_symbol=eos_idx, device=hyperparams["device"]
            )

            # Calculate accuracy
            token_acc, seq_acc = calculate_accuracy(pred, tgt, pad_idx)
            token_accs.append(token_acc)
            seq_accs.append(seq_acc)

    # Return the best (maximum) token and sequence accuracy
    return max(token_accs), max(seq_accs)

def print_results_and_plot(results):
    """Print summary of results and generate accuracy graphs."""
    sizes = []
    best_token_accuracies = []
    token_stds = []
    best_seq_accuracies = []
    seq_stds = []

    # Prepare data for the graph
    for size, accuracies in results.items():
        token_accuracies = [acc[0] for acc in accuracies]
        seq_accuracies = [acc[1] for acc in accuracies]

        # Best accuracies
        best_token_accuracy = max(token_accuracies)
        best_seq_accuracy = max(seq_accuracies)

        # Standard deviations
        token_std = np.std(token_accuracies)
        seq_std = np.std(seq_accuracies)

        best_token_accuracies.append(best_token_accuracy * 100)  # Convert to percentage
        token_stds.append(token_std * 100)  # Convert to percentage
        best_seq_accuracies.append(best_seq_accuracy * 100)  # Convert to percentage
        seq_stds.append(seq_std * 100)  # Convert to percentage
        sizes.append(size)  # Dataset size

        print(f"{size:10} | Best Token Acc: {best_token_accuracy * 100:.2f}% | Token Std: {token_std * 100:.2f}%")
        print(f"             | Best Seq Acc: {best_seq_accuracy * 100:.2f}% | Seq Std: {seq_std * 100:.2f}%")
        print("-" * 100)

    # Create positions for the bars
    x = np.arange(len(sizes))
    width = 0.35  # Bar width

    # Plotting the Token-Level Accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, best_token_accuracies, width, yerr=token_stds, capsize=5,
            label="Best Token Accuracy", color="teal", alpha=0.8)
    plt.bar(x + width / 2, best_seq_accuracies, width, yerr=seq_stds, capsize=5,
            label="Best Sequence Accuracy", color="gold", alpha=0.8)

    # Add labels and legend
    plt.xlabel("Number of Composed Commands Used for Training", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Best Token and Sequence Accuracy with Standard Deviation", fontsize=16)
    plt.xticks(x, sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and show plot
    plt.tight_layout()
    plt.savefig("experiment3_best_accuracy_plot.png")
    plt.show()

def run_experiment_3(n_runs=5):
    """Run Experiment 3: Adding a new primitive and testing generalization."""
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

    # Fetch dataset pairs
    pairs = get_add_prim_dataset_pairs()

    results = {}

    # Process all datasets
    for train_test_pairs, name in pairs:
        print(f"\nProcessing dataset {name}")
        print("=" * 70)

        num_results = []
        for train_path, test_path in train_test_pairs:
            for run in range(n_runs):
                seed = 42 + run
                print(f"Run {run + 1}/{n_runs} with seed {seed}")

                # Train the model
                model, _, _ = train(
                    train_path=train_path,
                    test_path=test_path,  # Validation during training
                    hyperparams=hyperparams,
                    model_suffix=name,
                    random_seed=seed,
                )

                # Load test dataset for evaluation
                test_data = SCANDataset(test_path)
                test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False)

                # Evaluate the model
                token_acc, seq_acc = evaluate(model, test_loader, hyperparams)
                num_results.append((token_acc, seq_acc))

                print(f"Run {run + 1} | Best Token Accuracy: {token_acc * 100:.2f}% | Best Sequence Accuracy: {seq_acc * 100:.2f}%")

        results[name] = num_results

    # Print summary of results and plot graphs
    print_results_and_plot(results)

if __name__ == "__main__":
    run_experiment_3()
