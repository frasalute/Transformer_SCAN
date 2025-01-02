import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SCANDataset
from transformer import Transformer
from tqdm import tqdm
from pathlib import Path

def greedy_decode(model, src, max_len, start_symbol, end_symbol, device):
    """Greedy decoding for autoregressive generation"""
    model.eval()
    src = src.to(device)
    
    # Initialize with start symbol
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type(torch.long).to(device)
    finished = torch.zeros(src.shape[0], dtype=torch.bool).to(device)
    
    for i in range(max_len-1):
        out = model(src, ys)
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        
        # Update sequences that are not finished
        ys = torch.cat([ys, next_word * ~finished.unsqueeze(1)], dim=1)
        
        # Update finished mask
        finished = finished | (next_word == end_symbol).squeeze(1)
        
        # Stop if all sequences have end symbol
        if finished.all():
            break
            
    return ys

def calculate_accuracy(pred, target, pad_idx):
    """Calculate token and sequence accuracy"""
    batch_size = pred.size(0)
    
    # Get max length and pad if needed
    max_len = max(pred.size(1), target.size(1))
    if pred.size(1) < max_len:
        pad_size = (batch_size, max_len - pred.size(1))
        pred = torch.cat([pred, torch.full(pad_size, pad_idx).to(pred.device)], dim=1)
    elif target.size(1) < max_len:
        pad_size = (batch_size, max_len - target.size(1))
        target = torch.cat([target, torch.full(pad_size, pad_idx).to(target.device)], dim=1)
    
    # Sequence length accuracy
    pred_lengths = (pred != pad_idx).sum(dim=1)
    target_lengths = (target != pad_idx).sum(dim=1)
    seq_acc = (pred_lengths == target_lengths).float().mean().item()

    # Reshape to 1D
    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)
    
    # Mask out padding tokens
    mask = target != pad_idx
    correct = (pred[mask] == target[mask]).float()
    token_acc = correct.mean().item()
    
    return token_acc, seq_acc

def evaluate(model, data_loader, criterion, pad_idx, device):
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            
            output = output.contiguous().view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

            # Calculate accuracies
            pred = output.argmax(dim=-1).view(tgt.size(0), -1)
            token_acc, seq_acc = calculate_accuracy(pred, tgt[:, 1:], pad_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_loss = total_loss / len(data_loader)
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc

def train(train_path, test_path, hyperparams, model_suffix, random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True)

    model = Transformer(
        src_vocab_size=train_dataset.src_vocab.vocab_size,
        tgt_vocab_size=train_dataset.tgt_vocab.vocab_size,
        src_pad_idx=train_dataset.src_vocab.special_tokens["<PAD>"],
        tgt_pad_idx=train_dataset.tgt_vocab.special_tokens["<PAD>"],
        emb_dim=hyperparams["emb_dim"],
        num_layers=hyperparams["n_layers"],
        num_heads=hyperparams["n_heads"],
        forward_dim=hyperparams["forward_dim"],
        dropout=hyperparams["dropout"],
    ).to(hyperparams["device"])

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tgt_vocab.special_tokens["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    best_acc = 0.0
    pad_idx = train_dataset.src_vocab.special_tokens["<PAD>"]
    bos_idx = train_dataset.src_vocab.special_tokens["<BOS>"]
    eos_idx = train_dataset.src_vocab.special_tokens["<EOS>"]

    print(f"Training for {hyperparams['epochs']} epochs")
    for epoch in range(hyperparams["epochs"]):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{hyperparams["epochs"]} [Train]')
        for batch in pbar:
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)

            output = output.contiguous().view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss, avg_token_acc, avg_seq_acc = evaluate(
            model, test_loader, criterion, pad_idx, hyperparams["device"]
        )

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Token Accuracy: {avg_token_acc:.4f}")
        print(f"Sequence Accuracy: {avg_seq_acc:.4f}")

        if avg_seq_acc > best_acc:
            best_acc = avg_seq_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparams': hyperparams,
                'accuracy': best_acc,
            }, CHECKPOINT_DIR / f'best_model_{model_suffix}.pt')

  

    # Final evaluation with greedy decode
    print("\nFinal Evaluation with Greedy Decode:")
    model.eval()
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Final Evaluation'):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])
            
            pred = greedy_decode(
                model, src, 
                max_len=tgt.size(1),
                start_symbol=bos_idx,
                end_symbol=eos_idx,
                device=hyperparams["device"]
            )
            
            token_acc, seq_acc = calculate_accuracy(pred[:, 1:], tgt[:, 1:], pad_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)
    
    print(f"Final Token Accuracy: {avg_token_acc:.4f}")
    print(f"Final Sequence Accuracy: {avg_seq_acc:.4f}")
    return model, avg_token_acc, avg_seq_acc


