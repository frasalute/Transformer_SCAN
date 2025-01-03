

import torch
import numpy as np
from train_greedsearch import main 

def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path  = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs


def run_all_variations(n_runs=1):
    """Run training multiple times for all dataset size variations with different seeds."""
    results = {}
    # Hyperparams for all experiments
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 1,
        "n_heads": 8,
        "forward_dim": 512,
        "dropout": 0.05,
        "learning_rate": 7e-4,
        "batch_size": 64,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Collect p{size} => list of (token_acc, seq_acc) for each run
    for _, _, size in get_dataset_pairs():
        results[f"p{size}"] = []

    for run_idx in range(n_runs):
        seed = 42 + run_idx
        print(f"\nStarting run {run_idx+1}/{n_runs} with seed {seed}")
        print("=" * 70)

        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            # Call 'main' => returns (model, final_tok_acc, final_seq_acc)
            _, token_acc, seq_acc = main(
                train_path=train_path, 
                test_path=test_path,
                model_suffix=f"p_{size}",
                hyperparams=hyperparams,
                random_seed=seed
            )
            results[f"p{size}"].append((token_acc, seq_acc))

    # Summarize results
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Token Acc ± Std | Mean Sequence Acc ± Std")
    print("-" * 50)

    for size, acc_list in results.items():
        token_accs = [acc_tuple[0] for acc_tuple in acc_list]
        seq_accs   = [acc_tuple[1] for acc_tuple in acc_list]

        mean_token = np.mean(token_accs)
        std_token  = np.std(token_accs)
        mean_seq   = np.mean(seq_accs)
        std_seq    = np.std(seq_accs)

        print(f"{size:11} | {mean_token:.4f} ± {std_token:.4f} | {mean_seq:.4f} ± {std_seq:.4f}")
        print(f"TokenAcc runs: {[f'{a:.4f}' for a in token_accs]}")
        print(f"SeqAcc   runs: {[f'{a:.4f}' for a in seq_accs]}")
        print()

if __name__ == "__main__":
    run_all_variations(n_runs=1)
