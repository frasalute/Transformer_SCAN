import matplotlib.pyplot as plt
import torch
from train_greedysearch import train, greedy_decode, calculate_accuracy
from dataset import SCANDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def evaluate_length_split(model, data_loader, hyperparams, oracle=False):
    """Evaluate the model on token-level and sequence-level accuracy."""
    pad_idx = data_loader.dataset.vocab.special_tokens["<PAD>"]
    bos_idx = data_loader.dataset.vocab.special_tokens["<BOS>"]
    eos_idx = data_loader.dataset.vocab.special_tokens["<EOS>"]

    results = {
        "action_lengths": {},
        "command_lengths": {},
    }
    token_accs = []
    seq_accs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            if oracle:
                pred = tgt  # Oracle decoding assumes target as predictions
            else:
                pred = greedy_decode(
                    model, src, max_len=tgt.size(1), start_symbol=bos_idx, end_symbol=eos_idx, device=hyperparams["device"]
                )

            token_acc, seq_acc = calculate_accuracy(pred, tgt, pad_idx)
            token_accs.append(token_acc)
            seq_accs.append(seq_acc)

            for i in range(src.size(0)):
                act_len = torch.count_nonzero(tgt[i]).item()
                cmd_len = torch.count_nonzero(src[i]).item()

                if act_len not in results["action_lengths"]:
                    results["action_lengths"][act_len] = {"accuracy": 0, "count": 0}
                if cmd_len not in results["command_lengths"]:
                    results["command_lengths"][cmd_len] = {"accuracy": 0, "count": 0}

                results["action_lengths"][act_len]["accuracy"] += seq_acc
                results["action_lengths"][act_len]["count"] += 1

                results["command_lengths"][cmd_len]["accuracy"] += seq_acc
                results["command_lengths"][cmd_len]["count"] += 1

    # Calculate average accuracy
    for length_data in [results["action_lengths"], results["command_lengths"]]:
        for length, data in length_data.items():
            data["accuracy"] /= data["count"]

    return results, np.mean(token_accs), np.mean(seq_accs)


def plot_results(results, filename_prefix, title_suffix):
    """Plot results for action lengths and command lengths."""
    action_lengths = list(results["action_lengths"].keys())
    action_accuracies = [results["action_lengths"][length]["accuracy"] * 100 for length in action_lengths]

    command_lengths = list(results["command_lengths"].keys())
    command_accuracies = [results["command_lengths"][length]["accuracy"] * 100 for length in command_lengths]

    # Plot accuracy by action sequence length
    plt.figure(figsize=(12, 6))
    plt.bar(action_lengths, action_accuracies, alpha=0.7, color="blue")
    plt.title(f"Token-Level Accuracy by Action Sequence Length {title_suffix}")
    plt.xlabel("Ground-Truth Action Sequence Length (in words)")
    plt.ylabel("Accuracy on New Commands (%)")
    plt.savefig(f"{filename_prefix}_action_lengths.png")
    plt.close()

    # Plot accuracy by command sequence length
    plt.figure(figsize=(12, 6))
    plt.bar(command_lengths, command_accuracies, alpha=0.7, color="blue")
    plt.title(f"Token-Level Accuracy by Command Length {title_suffix}")
    plt.xlabel("Command Length (in words)")
    plt.ylabel("Accuracy on New Commands (%)")
    plt.savefig(f"{filename_prefix}_command_lengths.png")
    plt.close()


def run_experiment():
    """Run Experiment 2 with greedy and oracle decoding."""
    # Initialize hyperparameters
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

    train_path = "/teamspace/studios/this_studio/Transformer_SCAN/data/length_split/tasks_train_length.txt"
    test_path = "/teamspace/studios/this_studio/Transformer_SCAN/data/length_split/tasks_test_length.txt"
    model_suffix = "experiment_2"
    random_seed = 42

    # Train the model
    print("Training the model...")
    model, _, _ = train(
        train_path=train_path,
        test_path=test_path,
        hyperparams=hyperparams,
        model_suffix=model_suffix,
        random_seed=random_seed,
    )

    # Load test dataset
    test_data = SCANDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False)

    # Evaluate without Oracle lengths
    print("\nEvaluating without Oracle lengths...")
    results_no_oracle, token_acc_no_oracle, seq_acc_no_oracle = evaluate_length_split(
        model, test_loader, hyperparams, oracle=False
    )

    # Evaluate with Oracle lengths
    print("\nEvaluating with Oracle lengths...")
    results_oracle, token_acc_oracle, seq_acc_oracle = evaluate_length_split(
        model, test_loader, hyperparams, oracle=True
    )

    # Print results
    print(f"\nWithout Oracle Length: Sequence-level accuracy: {seq_acc_no_oracle * 100:.2f}%, Token-level accuracy: {token_acc_no_oracle * 100:.2f}%")
    print(f"With Oracle Length: Sequence-level accuracy: {seq_acc_oracle * 100:.2f}%, Token-level accuracy: {token_acc_oracle * 100:.2f}%")

    # Plot results
    plot_results(results_no_oracle, filename_prefix="experiment2_no_oracle", title_suffix="Without Oracle Lengths")
    plot_results(results_oracle, filename_prefix="experiment2_oracle", title_suffix="With Oracle Lengths")


if __name__ == "__main__":
    run_experiment()
