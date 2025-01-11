import numpy as np
import torch
from train_greedysearch import train, greedy_decode, calculate_accuracy
from dataset import SCANDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print
from rich.traceback import install

install()


def get_add_prim_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3."""
    base_path = "/work/Transformer_SCAN/data/add_prim_split"

    # Basic datasets
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

    return np.mean(token_accs), np.mean(seq_accs)


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

    # Process the basic `jump` and `turn_left` datasets
    for train_path, test_path, name in pairs[:2]:
        print(f"\nProcessing dataset {name}")
        print("=" * 70)

        basic_results = []
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
            basic_results.append((token_acc, seq_acc))

            print(f"Run {run + 1} | Token Accuracy: {token_acc * 100:.2f}% | Sequence Accuracy: {seq_acc * 100:.2f}%")

        results[name] = basic_results

    # Process numerical datasets
    for train_test_pairs, num in pairs[2:]:
        print(f"\nProcessing dataset {num}")
        print("=" * 70)

        num_results = []
        for train_path, test_path in train_test_pairs:
            print(f"Evaluating repetition dataset: {train_path}")
            model, _, _ = train(
                train_path=train_path,
                test_path=test_path,
                hyperparams=hyperparams,
                model_suffix=num,
                random_seed=42,  # Fixed seed for numerical datasets
            )

            test_data = SCANDataset(test_path)
            test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False)

            token_acc, seq_acc = evaluate(model, test_loader, hyperparams)
            num_results.append((token_acc, seq_acc))

        results[num] = num_results

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
        print("-" * 50)


if __name__ == "__main__":
    run_experiment_3()
