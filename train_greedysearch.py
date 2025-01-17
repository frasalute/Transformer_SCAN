import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import SCANDataset
from transformer import Transformer
from tqdm import tqdm
from pathlib import Path


def greedy_decode(model, src, max_len, start_symbol, end_symbol, device):
    """Greedy decoding for autoregressive generation."""
    model.eval()
    src = src.to(device)

    ys = torch.ones(src.size(0), 1).fill_(start_symbol).long().to(device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        out = model(src, ys)  
        probs = out[:, -1, :]  
        next_word = probs.argmax(dim=-1, keepdim=True)  

        next_word = torch.where(finished.unsqueeze(1), ys[:, -1:], next_word)
        ys = torch.cat([ys, next_word], dim=1)

        is_eos = next_word.squeeze(1) == end_symbol
        finished |= is_eos

        if finished.all():
            break

    for i in range(ys.size(0)):
        eos_indices = (ys[i] == end_symbol).nonzero(as_tuple=True)[0]
        if eos_indices.numel() > 0:  
            ys[i, eos_indices[0] + 1:] = end_symbol  

    return ys


def calculate_accuracy(pred, target, pad_idx):
    """Calculate token-level and sequence-level accuracy."""
    batch_size = pred.size(0)

    # Ensure both are the same length by padding
    max_len = max(pred.size(1), target.size(1))
    if pred.size(1) < max_len:
        pred = torch.cat([pred, torch.full((batch_size, max_len - pred.size(1)), pad_idx, device=pred.device)], dim=1)
    elif target.size(1) < max_len:
        target = torch.cat([target, torch.full((batch_size, max_len - target.size(1)), pad_idx, device=target.device)], dim=1)

    
    mask = target != pad_idx
    token_acc = (pred[mask] == target[mask]).float().mean().item()

    
    seq_acc = ((pred == target) | ~mask).all(dim=1).float().mean().item()

    return token_acc, seq_acc


def evaluate(model, data_loader, criterion, pad_idx, device):
    """Evaluate the model using teacher forcing."""
    model.eval()
    total_loss = 0.0
    token_accs = []
    seq_accs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            out = model(src, tgt_input)  # Teacher forcing
            out = out.reshape(-1, out.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(out, tgt_output)
            total_loss += loss.item()

            
            pred = out.argmax(dim=-1).view(tgt.size(0), -1)
            tok_acc, seq_acc = calculate_accuracy(pred, tgt[:, 1:], pad_idx)
            token_accs.append(tok_acc)
            seq_accs.append(seq_acc)

    avg_loss = total_loss / len(data_loader)
    avg_token_acc = sum(token_accs) / len(token_accs)
    avg_seq_acc = sum(seq_accs) / len(seq_accs)

    return avg_loss, avg_token_acc, avg_seq_acc


def train(train_path, test_path, hyperparams, model_suffix, random_seed):
    """Train the Transformer on SCAN with teacher forcing and final greedy decode."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    
    train_data = SCANDataset(train_path)
    test_data = SCANDataset(test_path)

    train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True)

    # Create the Transformer model
    model = Transformer(
        src_vocab_size=train_data.vocab.vocab_size,
        tgt_vocab_size=train_data.vocab.vocab_size,
        src_pad_idx=train_data.vocab.special_tokens["<PAD>"],
        tgt_pad_idx=train_data.vocab.special_tokens["<PAD>"],
        emb_dim=hyperparams["emb_dim"],
        num_layers=hyperparams["n_layers"],
        num_heads=hyperparams["n_heads"],
        forward_dim=hyperparams["forward_dim"],
        dropout=hyperparams["dropout"],
    ).to(hyperparams["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_data.vocab.special_tokens["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"])

    pad_idx = train_data.vocab.special_tokens["<PAD>"]
    bos_idx = train_data.vocab.special_tokens["<BOS>"]
    eos_idx = train_data.vocab.special_tokens["<EOS>"]

    best_seq_acc = 0.0

    print(f"Training for {hyperparams['epochs']} epochs.")
    for epoch in range(hyperparams["epochs"]):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Train]")
        for batch in pbar:
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            out = model(src, tgt_input)
            out = out.reshape(-1, out.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(out, tgt_output)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate after each epoch
        val_loss, val_token_acc, val_seq_acc = evaluate(model, test_loader, criterion, pad_idx, hyperparams["device"])
        print(f"\nEpoch {epoch+1} Validation:")
        print(f"Loss: {val_loss:.4f} | Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f}")

        if val_seq_acc > best_seq_acc:
            best_seq_acc = val_seq_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hyperparams": hyperparams,
                    "accuracy": best_seq_acc,
                },
                CHECKPOINT_DIR / f"best_model_{model_suffix}.pt",
            )

    # Final evaluation with greedy decoding
    print("\nFinal Greedy Decode Evaluation:")
    model.eval()
    token_accs = []
    seq_accs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Greedy Evaluation"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            pred = greedy_decode(model, src, max_len=tgt.size(1), start_symbol=bos_idx, end_symbol=eos_idx, device=hyperparams["device"])
            tok_acc, seq_acc = calculate_accuracy(pred[:, 1:], tgt[:, 1:], pad_idx)
            token_accs.append(tok_acc)
            seq_accs.append(seq_acc)

    final_token_acc = sum(token_accs) / len(token_accs)
    final_seq_acc = sum(seq_accs) / len(seq_accs)

    print(f"Final Greedy Decode => Token Acc: {final_token_acc:.4f} | Seq Acc: {final_seq_acc:.4f}")

    return model, final_token_acc, final_seq_acc
