import matplotlib.pyplot as plt
import torch
from train_greedysearch import train, greedy_decode, calculate_accuracy
from dataset import SCANDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def truncate_and_pad(sequence, eos_idx, max_len, pad_idx):
    """Truncate a sequence at the first occurrence of <EOS> and pad it to max_len."""
    eos_positions = (sequence == eos_idx).nonzero(as_tuple=True)[0]
    if eos_positions.numel() > 0:  
        sequence = sequence[: eos_positions[0] + 1]
    padded_sequence = torch.full((max_len,), pad_idx, device=sequence.device)
    padded_sequence[:sequence.size(0)] = sequence
    return padded_sequence

def oracle_decode(model, src, tgt, bos_idx, eos_idx, pad_idx, max_len, device):
    """Generate oracle-constrained predictions for evaluation."""
    model.eval()
    batch_size = src.size(0)
    encode_out = model.encoder(src, model.create_src_mask(src))
    pred = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    
    min_lengths = torch.tensor(
        [
            ((seq == eos_idx).nonzero(as_tuple=True)[0].item() + 1)
            if (seq == eos_idx).any()
            else seq.size(0)
            for seq in tgt
        ],
        device=device,
    )

    for _ in range(max_len - 1):
        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(pred, encode_out, model.create_src_mask(src), tgt_mask)
        logits = decode_out[:, -1, :]

        # Prevent early EOS
        current_len = pred.size(1)
        mask = current_len < min_lengths
        logits[mask, eos_idx] = float("-inf")

        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        next_token = next_token.masked_fill(finished.unsqueeze(1), pad_idx)

        pred = torch.cat([pred, next_token], dim=1)
        newly_finished = next_token.squeeze(1) == eos_idx
        finished = finished | newly_finished

        if finished.all():
            break

    return pred

def evaluate_length_split(model, data_loader, hyperparams, oracle=False):
    pad_idx = data_loader.dataset.vocab.special_tokens["<PAD>"]
    bos_idx = data_loader.dataset.vocab.special_tokens["<BOS>"]
    eos_idx = data_loader.dataset.vocab.special_tokens["<EOS>"]

    results = {
        "action_lengths": {},
        "command_lengths": {},
    }

    total_seq_matches = 0
    total_seq_count = 0
    total_token_correct = 0
    total_token_count = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            # Oracle or greedy decode
            if oracle:
                pred = oracle_decode(
                    model,
                    src,
                    tgt,
                    bos_idx,
                    eos_idx,
                    pad_idx,
                    max_len=tgt.size(1),
                    device=hyperparams["device"],
                )
            else:
                pred = greedy_decode(
                    model, 
                    src, 
                    max_len=tgt.size(1), 
                    start_symbol=bos_idx, 
                    end_symbol=eos_idx, 
                    device=hyperparams["device"],
                )

            
            pred_trunc = torch.stack(
                [truncate_and_pad(p, eos_idx, tgt.size(1), pad_idx) for p in pred]
            )
            tgt_trunc = torch.stack(
                [truncate_and_pad(t, eos_idx, tgt.size(1), pad_idx) for t in tgt]
            )

            
            mask = (tgt_trunc != pad_idx)
            exact_matches = ((pred_trunc == tgt_trunc) | ~mask).all(dim=1).float()  

            
            correct_tokens = (pred_trunc[mask] == tgt_trunc[mask]).sum().item()
            total_tokens = mask.sum().item()

            
            total_seq_matches += exact_matches.sum().item()
            total_seq_count += exact_matches.size(0)
            total_token_correct += correct_tokens
            total_token_count += total_tokens

            
            for i in range(src.size(0)):
                act_len = (tgt[i] != pad_idx).sum().item()
                cmd_len = (src[i] != pad_idx).sum().item()

                if act_len not in results["action_lengths"]:
                    results["action_lengths"][act_len] = {"accuracy": 0.0, "count": 0}
                if cmd_len not in results["command_lengths"]:
                    results["command_lengths"][cmd_len] = {"accuracy": 0.0, "count": 0}

                results["action_lengths"][act_len]["accuracy"] += exact_matches[i].item()
                results["action_lengths"][act_len]["count"] += 1

                results["command_lengths"][cmd_len]["accuracy"] += exact_matches[i].item()
                results["command_lengths"][cmd_len]["count"] += 1

    
    avg_seq_acc = total_seq_matches / total_seq_count
    avg_token_acc = total_token_correct / total_token_count

    
    for length_data in [results["action_lengths"], results["command_lengths"]]:
        for length, data in length_data.items():
            data["accuracy"] /= data["count"]

    return results, avg_token_acc, avg_seq_acc
    
def run_experiment():
    """Run Experiment 2 with greedy and oracle decoding."""
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

    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    model_suffix = "experiment_2"
    random_seed = 42

    print("Training the model...")
    model, _, _ = train(
        train_path=train_path,
        test_path=test_path,
        hyperparams=hyperparams,
        model_suffix=model_suffix,
        random_seed=random_seed,
    )

    
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

    print(f"\nWithout Oracle Length: Sequence-level accuracy: {seq_acc_no_oracle * 100:.2f}%, Token-level accuracy: {token_acc_no_oracle * 100:.2f}%")
    print(f"With Oracle Length: Sequence-level accuracy: {seq_acc_oracle * 100:.2f}%, Token-level accuracy: {token_acc_oracle * 100:.2f}%")

if __name__ == "__main__":
    run_experiment()