from t5_dataset import SCANDataset
from t5_model_large import fine_tune_t5
from transformers import T5Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs

def run_experiment():
    """Run Experiment 1."""
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)  # tried for both large and base
    results = {}
    hyperparams = {
        "learning_rate": 1e-5,  
        "batch_size": 8, 
        "epochs": 20,  
        "gradient_accumulation_steps": 8,  
        "weight_decay": 1e-3,  
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }  # these hyperparameters ar very low, they don't overload memory but at the same time don't actually allow the model to learn

    for train_path, test_path, size in get_dataset_pairs():
        print(f"Running for dataset size p{size}...")
        train_dataset = SCANDataset(train_path, tokenizer)  
        test_dataset = SCANDataset(test_path, tokenizer)  
        _, token_acc, seq_acc = fine_tune_t5(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            hyperparams=hyperparams,
            model_suffix=f"p{size}",
            random_seed=42,
            tokenizer=tokenizer  
        )
        results[size] = (token_acc, seq_acc)

    # Visualization
    sizes = [int(size) for size in results.keys()]
    token_accs = [results[size][0] for size in results]
    seq_accs = [results[size][1] for size in results]

    bar_width = 0.35  
    x_indices = np.arange(len(sizes))  

    plt.figure(figsize=(10, 6))
    plt.bar(x_indices - bar_width / 2, token_accs, width=bar_width, label="Token Accuracy", color="teal")
    plt.bar(x_indices + bar_width / 2, seq_accs, width=bar_width, label="Sequence Accuracy", color="gold")

    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Accuracy")
    plt.title("Token and Sequence Accuracy")
    plt.xticks(x_indices, sizes)  
    plt.legend()
    plt.grid(axis="y")  

    output_path = "experiment1_results_histogram.png"
    plt.savefig(output_path, format="png", dpi=300)
    print(f"Figure saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    run_experiment()