from t5_dataset import SCANDataset
from t5_model import fine_tune_t5
from transformers import T5Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "/work/Transformer_SCAN/data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs

def run_experiment():
    """Run Experiment 1."""
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Load the tokenizer
    results = {}
    hyperparams = {
        "learning_rate": 5e-5,
        "batch_size": 8,
        "epochs": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    for train_path, test_path, size in get_dataset_pairs():
        print(f"Running for dataset size p{size}...")
        train_dataset = SCANDataset(train_path, tokenizer)  # Pass tokenizer to SCANDataset
        test_dataset = SCANDataset(test_path, tokenizer)  # Pass tokenizer to SCANDataset
        _, token_acc, seq_acc = fine_tune_t5(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            hyperparams=hyperparams,
            model_suffix=f"p{size}",
            random_seed=42,
            tokenizer=tokenizer  # Pass tokenizer to fine_tune_t5
        )
        results[size] = (token_acc, seq_acc)

    # Visualization
    sizes = [int(size) for size in results.keys()]
    token_accs = [results[size][0] for size in results]
    seq_accs = [results[size][1] for size in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, token_accs, label="Token Accuracy", marker="o")
    plt.plot(sizes, seq_accs, label="Sequence Accuracy", marker="s")
    plt.xlabel("Training Dataset Size (%)")
    plt.ylabel("Accuracy")
    plt.title("Token and Sequence Accuracy vs Dataset Size")
    plt.legend()
    plt.grid()

    output_path = "experiment_results.png"  
    plt.savefig(output_path, format="png", dpi=300)
    print(f"Figure saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    run_experiment()
