import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from t5_dataset import SCANDataset
from t5_model import fine_tune_t5
import matplotlib.pyplot as plt

def run_experiment_2():
    """Run Experiment 2: Length Generalization."""
    # Hyperparameters
    hyperparams = {
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Dataset paths
    train_path = "/work/Transformer_SCAN/data/length_split/tasks_train_length.txt"
    test_path = "/work/Transformer_SCAN/data/length_split/tasks_test_length.txt"

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load datasets
    train_dataset = SCANDataset(train_path, tokenizer)
    test_dataset = SCANDataset(test_path, tokenizer)

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

    # Evaluate the model
    evaluate_and_plot(test_dataset, model, tokenizer, hyperparams)

def evaluate_and_plot(test_dataset, model, tokenizer, hyperparams):
    """Evaluate the model and generate plots for Experiment 2."""
    # Prepare dictionaries to store accuracies
    token_accuracies_by_action_length = {}
    token_accuracies_by_command_length = {}

    # Evaluation
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    print("Evaluating the model on the test dataset...")

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            # Generate predictions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, target, command in zip(predictions, targets, batch["input_ids"]):
                # Token accuracy
                token_acc = sum([p == t for p, t in zip(pred.split(), target.split())]) / len(target.split())
                command_length = len(tokenizer.decode(command, skip_special_tokens=True).split())
                action_length = len(target.split())

                # Debugging: Print details
                print(f"Prediction: {pred}")
                print(f"Target: {target}")
                print(f"Action Length: {action_length}, Command Length: {command_length}, Token Accuracy: {token_acc}")

                # Group by action length
                if action_length not in token_accuracies_by_action_length:
                    token_accuracies_by_action_length[action_length] = []
                token_accuracies_by_action_length[action_length].append(token_acc)

                # Group by command length
                if command_length not in token_accuracies_by_command_length:
                    token_accuracies_by_command_length[command_length] = []
                token_accuracies_by_command_length[command_length].append(token_acc)

    # Compute mean accuracies
    print("\nComputing mean accuracies...")
    token_accuracies_by_action_length = {
        length: np.mean(accs) for length, accs in token_accuracies_by_action_length.items()
    }
    token_accuracies_by_command_length = {
        length: np.mean(accs) for length, accs in token_accuracies_by_command_length.items()
    }

    # Debugging: Print grouped metrics
    print("Token Accuracies by Action Length:", token_accuracies_by_action_length)
    print("Token Accuracies by Command Length:", token_accuracies_by_command_length)

    # Plot the results
    plot_results(token_accuracies_by_action_length, token_accuracies_by_command_length)

def plot_results(action_accs, command_accs):
    """Plot token-level accuracy grouped by action and command lengths."""
    # Debugging: Ensure the data being plotted is valid
    print("\nPlotting results...")
    print("Action Length Keys:", list(action_accs.keys()))
    print("Action Length Values:", list(action_accs.values()))
    print("Command Length Keys:", list(command_accs.keys()))
    print("Command Length Values:", list(command_accs.values()))

    plt.figure(figsize=(12, 5))

    # Left plot: Accuracy by action sequence length
    plt.subplot(1, 2, 1)
    plt.bar(action_accs.keys(), action_accs.values(), color='blue', alpha=0.7)
    plt.xlabel("Ground-Truth Action Sequence Length (in words)")
    plt.ylabel("Token-Level Accuracy (%)")
    plt.title("Token-Level Accuracy by Action Sequence Length")
    plt.ylim(0, 100)

    # Right plot: Accuracy by command length
    plt.subplot(1, 2, 2)
    plt.bar(command_accs.keys(), command_accs.values(), color='orange', alpha=0.7)
    plt.xlabel("Command Length (in words)")
    plt.ylabel("Token-Level Accuracy (%)")
    plt.title("Token-Level Accuracy by Command Length")
    plt.ylim(0, 100)

    # Save and show the plot
    plt.suptitle("Experiment 2: Token-Level Results")
    plt.tight_layout()
    plt.savefig("experiment_2_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_experiment_2()
