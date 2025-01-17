import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from t5_dataset import SCANDataset
from adafactort5_model import fine_tune_t5
import matplotlib.pyplot as plt

def run_experiment_2():
    """Run Experiment 2: Length Generalization."""
    # Hyperparameters
    hyperparams = {
        "learning_rate": 1e-4,  
        "batch_size": 64,      
        "max_length": 128,     
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Dataset paths
    train_path = "/work/Transformer_SCAN/data/length_split/tasks_train_length.txt"
    test_path = "/work/Transformer_SCAN/data/length_split/tasks_test_length.txt"

    # Load tokenizer (T5-small)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load datasets
    train_dataset = SCANDataset(train_path, tokenizer)
    test_dataset  = SCANDataset(test_path,  tokenizer)

    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    # Fine-tune the model
    model, token_acc, seq_acc = fine_tune_t5(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        hyperparams=hyperparams,
        model_suffix="length_split",
        random_seed=42,
        tokenizer=tokenizer
    )

    # Evaluate
    evaluate_and_plot(test_dataset, model, tokenizer, hyperparams)

def evaluate_and_plot(test_dataset, model, tokenizer, hyperparams):
    """
    Evaluate the model and generate plots for Experiment 2,
    including both token-level and sequence-level accuracy.
    We also compute the overall mean token and sequence accuracy.
    """

    # Dicts to store token accuracies by action/command length
    token_accuracies_by_action_length = {}
    token_accuracies_by_command_length = {}

    # Dicts to store sequence accuracies by action/command length
    sequence_accuracies_by_action_length = {}
    sequence_accuracies_by_command_length = {}

    # --- Overall accuracies across *all* test examples ---
    all_token_accs = []
    all_seq_accs   = []

    model.eval()
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=hyperparams["batch_size"], 
                                 shuffle=False)

    print("\nEvaluating the model on the test dataset...")

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50
            )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, target, command in zip(predictions, targets, batch["input_ids"]):
                target_tokens = target.split()
                pred_tokens   = pred.split()

                # Token-level accuracy
                if len(target_tokens) > 0:
                    token_acc = sum(p == t for p, t in zip(pred_tokens, target_tokens)) / len(target_tokens)
                else:
                    token_acc = 0.0

                # Sequence-level accuracy (exact match)
                seq_acc = 1.0 if pred == target else 0.0

                # Accumulate for overall mean
                all_token_accs.append(token_acc)
                all_seq_accs.append(seq_acc)

                # Get lengths
                command_length = len(tokenizer.decode(command, skip_special_tokens=True).split())
                action_length  = len(target_tokens)

                # Debug
                print(f"Prediction: {pred}")
                print(f"Target: {target}")
                print(f"Action Length: {action_length}, Command Length: {command_length}, "
                      f"Token Accuracy: {token_acc:.3f}, Sequence Accuracy: {seq_acc}")

                # Group token accuracy by action length
                token_accuracies_by_action_length.setdefault(action_length, []).append(token_acc)
                # Group token accuracy by command length
                token_accuracies_by_command_length.setdefault(command_length, []).append(token_acc)
                # Group sequence accuracy by action length
                sequence_accuracies_by_action_length.setdefault(action_length, []).append(seq_acc)
                # Group sequence accuracy by command length
                sequence_accuracies_by_command_length.setdefault(command_length, []).append(seq_acc)

    # --- Compute mean accuracies per length ---
    print("\nComputing mean accuracies by length groups...")
    token_accuracies_by_action_length = {
        length: np.mean(accs)
        for length, accs in token_accuracies_by_action_length.items()
    }
    token_accuracies_by_command_length = {
        length: np.mean(accs)
        for length, accs in token_accuracies_by_command_length.items()
    }
    sequence_accuracies_by_action_length = {
        length: np.mean(accs)
        for length, accs in sequence_accuracies_by_action_length.items()
    }
    sequence_accuracies_by_command_length = {
        length: np.mean(accs)
        for length, accs in sequence_accuracies_by_command_length.items()
    }

    # --- Compute *overall* mean accuracies ---
    overall_token_acc = np.mean(all_token_accs)
    overall_seq_acc   = np.mean(all_seq_accs)

    print("Token Accuracies by Action Length:", token_accuracies_by_action_length)
    print("Token Accuracies by Command Length:", token_accuracies_by_command_length)
    print("Sequence Accuracies by Action Length:", sequence_accuracies_by_action_length)
    print("Sequence Accuracies by Command Length:", sequence_accuracies_by_command_length)
    print("\n================= OVERALL TEST ACCURACY =================")
    print(f"Overall Mean Token Accuracy:    {overall_token_acc:.3f}")
    print(f"Overall Mean Sequence Accuracy: {overall_seq_acc:.3f}")

    # Plot the results
    plot_results(token_accuracies_by_action_length,
                 token_accuracies_by_command_length,
                 sequence_accuracies_by_action_length,
                 sequence_accuracies_by_command_length)

def plot_results(token_action_accs,
                 token_command_accs,
                 seq_action_accs,
                 seq_command_accs):
    """
    Plot token-level and sequence-level accuracy grouped by
    action length and command length.
    """

    print("\nPlotting results...")

    # Convert from fraction [0,1] to percent [0,100]
    token_action_vals = [v * 100 for v in token_action_accs.values()]
    token_command_vals = [v * 100 for v in token_command_accs.values()]
    seq_action_vals = [v * 100 for v in seq_action_accs.values()]
    seq_command_vals = [v * 100 for v in seq_command_accs.values()]

    # Sort keys
    sorted_action_lengths = sorted(set(token_action_accs.keys()) | set(seq_action_accs.keys()))
    sorted_command_lengths = sorted(set(token_command_accs.keys()) | set(seq_command_accs.keys()))

    token_action_vals_sorted = [token_action_accs[k]*100 for k in sorted_action_lengths]
    seq_action_vals_sorted   = [seq_action_accs[k]*100   for k in sorted_action_lengths]
    token_command_vals_sorted= [token_command_accs[k]*100 for k in sorted_command_lengths]
    seq_command_vals_sorted  = [seq_command_accs[k]*100   for k in sorted_command_lengths]

    print("Sorted Action Length Keys:", sorted_action_lengths)
    print("Sorted Action Length Token Acc Values:", token_action_vals_sorted)
    print("Sorted Action Length Sequence Acc Values:", seq_action_vals_sorted)
    print("Sorted Command Length Keys:", sorted_command_lengths)
    print("Sorted Command Length Token Acc Values:", token_command_vals_sorted)
    print("Sorted Command Length Sequence Acc Values:", seq_command_vals_sorted)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)

    # (1) Token Acc by Action Length
    axes[0, 0].bar(sorted_action_lengths, token_action_vals_sorted, color='teal', alpha=0.7)
    axes[0, 0].set_xlabel("Action Length (words)")
    axes[0, 0].set_ylabel("Token-Level Accuracy (%)")
    axes[0, 0].set_title("Token Acc by Action Length")
    axes[0, 0].set_ylim(0, 100)

    # (2) Token Acc by Command Length
    axes[0, 1].bar(sorted_command_lengths, token_command_vals_sorted, color='teal', alpha=0.7)
    axes[0, 1].set_xlabel("Command Length (words)")
    axes[0, 1].set_ylabel("Token-Level Accuracy (%)")
    axes[0, 1].set_title("Token Acc by Command Length")
    axes[0, 1].set_ylim(0, 100)

    # (3) Sequence Acc by Action Length
    axes[1, 0].bar(sorted_action_lengths, seq_action_vals_sorted, color='gold', alpha=0.7)
    axes[1, 0].set_xlabel("Action Length (words)")
    axes[1, 0].set_ylabel("Sequence-Level Accuracy (%)")
    axes[1, 0].set_title("Sequence Acc by Action Length")
    axes[1, 0].set_ylim(0, 100)

    # (4) Sequence Acc by Command Length
    axes[1, 1].bar(sorted_command_lengths, seq_command_vals_sorted, color='gold', alpha=0.7)
    axes[1, 1].set_xlabel("Command Length (words)")
    axes[1, 1].set_ylabel("Sequence-Level Accuracy (%)")
    axes[1, 1].set_title("Sequence Acc by Command Length")
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("experiment2_results_with_seq_acc.png", format="png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_experiment_2()


