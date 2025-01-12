from t5_dataset import SCANDataset
from t5_model import fine_tune_t5
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

def run_experiment_2():
    """Run Experiment 2: Length Generalization"""
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Load tokenizer

    # Load datasets
    train_dataset = SCANDataset(
        "/work/Transformer_SCAN/data/length_split/tasks_train_length.txt",
        tokenizer
    )
    test_dataset = SCANDataset(
        "/work/Transformer_SCAN/data/length_split/tasks_test_length.txt",
        tokenizer
    )

    # Hyperparameters
    hyperparams = {
        "learning_rate": 5e-5,
        "batch_size": 8,
        "epochs": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Fine-tune the model
    model, token_acc, seq_acc = fine_tune_t5(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        hyperparams=hyperparams,
        model_suffix="length_split",
        random_seed=42,
        tokenizer=tokenizer
    )

    print(f"Final Token Accuracy: {token_acc:.4f}, Final Sequence Accuracy: {seq_acc:.4f}")

    # Evaluation metrics by length
    token_accuracies_by_action_length = {}
    token_accuracies_by_command_length = {}

    with torch.no_grad():
        test_dataloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
        model.eval()

        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, target, command in zip(predictions, targets, batch["input_ids"]):
                token_acc = sum([p == t for p, t in zip(pred.split(), target.split())]) / len(target.split())
                command_length = len(tokenizer.decode(command, skip_special_tokens=True).split())
                action_length = len(target.split())

                # Update metrics by action length
                if action_length not in token_accuracies_by_action_length:
                    token_accuracies_by_action_length[action_length] = []
                token_accuracies_by_action_length[action_length].append(token_acc)

                # Update metrics by command length
                if command_length not in token_accuracies_by_command_length:
                    token_accuracies_by_command_length[command_length] = []
                token_accuracies_by_command_length[command_length].append(token_acc)

    # Compute mean accuracies
    token_accuracies_by_action_length = {
        length: np.mean(accs) for length, accs in token_accuracies_by_action_length.items()
    }
    token_accuracies_by_command_length = {
        length: np.mean(accs) for length, accs in token_accuracies_by_command_length.items()
    }

    # Plot results
    plt.figure(figsize=(12, 5))

    # Left plot: Accuracy by action sequence length
    plt.subplot(1, 2, 1)
    plt.bar(
        token_accuracies_by_action_length.keys(),
        token_accuracies_by_action_length.values(),
        color='blue', alpha=0.7
    )
    plt.xlabel("Ground-Truth Action Sequence Length (in words)")
    plt.ylabel("Accuracy on New Commands (%)")
    plt.title("Token-Level Accuracy by Action Sequence Length")
    plt.ylim(0, 100)

    # Right plot: Accuracy by command length
    plt.subplot(1, 2, 2)
    plt.bar(
        token_accuracies_by_command_length.keys(),
        token_accuracies_by_command_length.values(),
        color='orange', alpha=0.7
    )
    plt.xlabel("Command Length (in words)")
    plt.ylabel("Accuracy on New Commands (%)")
    plt.title("Token-Level Accuracy by Command Length")
    plt.ylim(0, 100)

    plt.suptitle("Experiment 2: Token-Level Results Without Oracle Lengths")
    plt.tight_layout()
    plt.savefig("experiment_2_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_experiment_2()
