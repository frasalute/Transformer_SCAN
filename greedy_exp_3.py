import numpy as np
import torch
from train_greedysearch import train, greedy_decode, calculate_accuracy
from dataset import SCANDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_add_prim_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3."""
    base_path = "data/add_prim_split"

    pairs = [
        (
            f"{base_path}/tasks_train_addprim_jump.txt",
            f"{base_path}/tasks_test_addprim_jump.txt",
            "jump",
        ),
        (
            f"{base_path}/tasks_train_addprim_turn_left.txt",
            f"{base_path}/tasks_test_addprim_turn_left.txt",
            "turn_left",
        ),
    ]

    # Adding additional splits (num1, num2, ..., num32)
    additional_base_path = "data/add_prim_split/with_additional_examples"
    num_composed_commands = ["num1", "num2", "num4", "num8", "num16", "num32"]
    for num in num_composed_commands:
        for rep in range(1, 2):  # Adjust repetition count as needed
            train_path = f"{additional_base_path}/tasks_train_addprim_complex_jump_{num}_rep{rep}.txt"
            test_path = f"{additional_base_path}/tasks_test_addprim_complex_jump_{num}_rep{rep}.txt"
            pairs.append((train_path, test_path, f"{num}_rep{rep}"))

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

    return np.mean(token_accs), np.mean(seq_accs)


def run_experiment_3(n_runs=1):
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

    for train_path, test_path, size in pairs:
        results[size] = []
        print(f"\nStarting training for dataset: {size}")
        print("=" * 70)

        for run in range(n_runs):
            seed = 42 + run  # Ensure reproducibility with different seeds
            print(f"Run {run + 1}/{n_runs} with seed {seed}")

            # Train the model
            model, _, _ = train(
                train_path=train_path,
                test_path=test_path,  # Validation during training
                hyperparams=hyperparams,
                model_suffix=size,
                random_seed=seed,
            )

            # Load test dataset for evaluation
            test_data = SCANDataset(test_path)
            test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False)

            # Evaluate using greedy decoding
            token_acc, seq_acc = evaluate(model, test_loader, hyperparams)
            results[size].append((token_acc, seq_acc))

            print(f"Run {run + 1} | Token Accuracy: {token_acc * 100:.2f}% | Sequence Accuracy: {seq_acc * 100:.2f}%")

    # Print summary of results
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset | Mean Token Accuracy ± Std Dev | Mean Seq Accuracy ± Std Dev")
    print("-" * 50)

    for size, accuracies in results.items():
        token_accuracies = [acc[0] for acc in accuracies]
        seq_accuracies = [acc[1] for acc in accuracies]

        token_mean, token_std = np.mean(token_accuracies), np.std(token_accuracies)
        seq_mean, seq_std = np.mean(seq_accuracies), np.std(seq_accuracies)

        print(f"{size:10} | {token_mean * 100:.2f} ± {token_std * 100:.2f} | {seq_mean * 100:.2f} ± {seq_std * 100:.2f}")
        print(f"Individual Token Accuracies: {', '.join(f'{acc * 100:.2f}' for acc in token_accuracies)}")
        print(f"Individual Sequence Accuracies: {', '.join(f'{acc * 100:.2f}' for acc in seq_accuracies)}")
        print("-" * 50)


if __name__ == "__main__":
    run_experiment_3()
