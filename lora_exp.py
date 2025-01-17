from t5_dataset import SCANDataset
from lora_model import fine_tune_t5_with_lora  
from transformers import T5Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_dataset_pairs():
    base_path = "/teamspace/studios/this_studio/Transformer_SCAN/data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs

def run_experiment():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  
    results = {}
    hyperparams = {
        "learning_rate": 5e-4,  
        "batch_size": 8,      
        "epochs": 30,          
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    for train_path, test_path, size in get_dataset_pairs():
        print(f"Running for dataset size p{size} with LoRA adaptation...")
        train_dataset = SCANDataset(train_path, tokenizer)
        test_dataset = SCANDataset(test_path, tokenizer)
        
        # Use LoRA fine-tuning
        _, token_acc, seq_acc = fine_tune_t5_with_lora(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            hyperparams=hyperparams,
            model_suffix=f"p{size}",
            random_seed=42,
            tokenizer=tokenizer
        )
        results[size] = (token_acc, seq_acc)

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
    plt.title("Token and Sequence Accuracy with LoRA")
    plt.xticks(x_indices, sizes)
    plt.legend()
    plt.grid(axis="y")

    output_path = "experiment1_results_histogram_lora.png"
    plt.savefig(output_path, format="png", dpi=300)
    print(f"Figure saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    run_experiment()
