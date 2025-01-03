import sys
import json
import matplotlib.pyplot as plt
sys.path.append('/Users/francescasalute/Dropbox/Mac/Documents/Master in Data Science/Third Semester/Advanced NLP/Transformer_SCAN')
from train_beamsearch import train, beam_search_decode, calculate_accuracy
import torch
import numpy as np
from dataset import SCANDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def save_results(results, filename="experiment2_results.json"):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

def plot_accuracy_per_length(results, filename_prefix="experiment2"):
    """Plot accuracy per command and action sequence length."""
    command_lengths = list(results["command_lengths"].keys())
    action_lengths = list(results["action_lengths"].keys())

    command_accuracies = [
        results["command_lengths"][length]["accuracy"]
        for length in command_lengths
    ]
    action_accuracies = [
        results["action_lengths"][length]["accuracy"]
        for length in action_lengths
    ]

    # Plot Token Accuracy per Command Sequence Length
    plt.figure(figsize=(10, 6))
    plt.bar(command_lengths, command_accuracies, color="blue", alpha=0.7)
    plt.xlabel("Command Sequence Length")
    plt.ylabel("Token Accuracy")
    plt.title("Token Accuracy per Command Sequence Length")
    plt.savefig(f"{filename_prefix}_command_length_accuracy.png")
    print(f"Command Length Accuracy graph saved to {filename_prefix}_command_length_accuracy.png")
    plt.close()

    # Plot Token Accuracy per Action Sequence Length
    plt.figure(figsize=(10, 6))
    plt.bar(action_lengths, action_accuracies, color="green", alpha=0.7)
    plt.xlabel("Action Sequence Length")
    plt.ylabel("Token Accuracy")
    plt.title("Token Accuracy per Action Sequence Length")
    plt.savefig(f"{filename_prefix}_action_length_accuracy.png")
    print(f"Action Length Accuracy graph saved to {filename_prefix}_action_length_accuracy.png")
    plt.close()

def analyze_per_length(results, data_loader, model, hyperparams):
    """Analyze accuracies per command and action sequence lengths."""
    pad_idx = data_loader.dataset.tgt_vocab.special_tokens["<PAD>"]
    bos_idx = data_loader.dataset.tgt_vocab.special_tokens["<BOS>"]
    eos_idx = data_loader.dataset.tgt_vocab.special_tokens["<EOS>"]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Final Evaluation"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            pred = beam_search_decode(
                model,
                src,
                max_len=tgt.size(1),
                start_symbol=bos_idx,
                end_symbol=eos_idx,
                beam_size=3,
                device=hyperparams["device"],
                pad_idx=pad_idx,
            )

            pred = pred[:, :tgt.size(1)]  # Truncate if necessary

            # Analyze accuracies per command and action sequence lengths
            for i in range(src.size(0)):
                cmd_len = torch.count_nonzero(src[i]).item()  # Command length
                act_len = torch.count_nonzero(tgt[i]).item()  # Action length
                _, seq_acc = calculate_accuracy(pred[i].unsqueeze(0), tgt[i].unsqueeze(0), pad_idx)

                if cmd_len not in results["command_lengths"]:
                    results["command_lengths"][cmd_len] = {"accuracy": 0, "count": 0}
                if act_len not in results["action_lengths"]:
                    results["action_lengths"][act_len] = {"accuracy": 0, "count": 0}

                results["command_lengths"][cmd_len]["accuracy"] += seq_acc
                results["command_lengths"][cmd_len]["count"] += 1
                results["action_lengths"][act_len]["accuracy"] += seq_acc
                results["action_lengths"][act_len]["count"] += 1

    # Calculate average accuracy per length
    for length_data in [results["command_lengths"], results["action_lengths"]]:
        for length, data in length_data.items():
            data["accuracy"] /= data["count"]

def run_experiment():
    """Run training for Experiment 2."""
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

    train_path = "Users/francescasalute/Dropbox/Mac/Documents/Master in Data Science/Third Semester/Advanced NLP/Transformer_SCAN/data/length_split/tasks_train_length.txt"
    test_path = "Users/francescasalute/Dropbox/Mac/Documents/Master in Data Science/Third Semester/Advanced NLP/Transformer_SCAN/data/length_split/tasks_test_length.txt"
    model_suffix = "length"

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Load the full test dataset and create a subset
    full_test_dataset = SCANDataset(test_path)
    subset_size = len(full_test_dataset) // 3  # Use 33% of the data
    test_subset = Subset(full_test_dataset, range(subset_size))

    # Create DataLoader for the test subset
    test_loader = DataLoader(
        test_subset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    # Run training
    print(f"Starting training for Experiment 2 (Length Split)")
    print("=" * 50)

    model, _, _ = train(
        train_path=train_path,
        test_path=test_path,  # Full test dataset used for training validation
        hyperparams=hyperparams,
        model_suffix=model_suffix,
        random_seed=seed,
    )

    # Analyze per-length accuracies
    results = {
        "command_lengths": {},
        "action_lengths": {},
    }
    analyze_per_length(results, test_loader, model, hyperparams)

    # Save results
    save_results(results)

    # Plot the graphs
    plot_accuracy_per_length(results)

if __name__ == "__main__":
    run_experiment()