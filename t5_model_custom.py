import torch
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration, AdamW
import numpy as np

def fine_tune_t5(train_dataset, test_dataset, hyperparams, model_suffix, random_seed, tokenizer):
    """Fine-tune the T5 model on the training dataset and evaluate on the test dataset."""
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Configure the T5 model
    config = T5Config(
        d_model=hyperparams["emb_dim"],          # EMB_DIM
        num_layers=hyperparams["n_layers"],     # N_LAYERS
        num_heads=hyperparams["n_heads"],       # N_HEADS
        d_ff=hyperparams["forward_dim"],        # FORWARD_DIM
        dropout_rate=hyperparams["dropout"]     # DROPOUT
    )
    model = T5ForConditionalGeneration(config)
    model.to(hyperparams["device"])

    # Prepare DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"])

    # Training loop
    model.train()
    for epoch in range(hyperparams["epochs"]):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{hyperparams['epochs']}, Loss: {total_loss / len(train_dataloader)}")

    # Evaluation
    model.eval()
    token_accuracies = []
    sequence_accuracies = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            # Generate predictions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, target in zip(predictions, targets):
                token_acc = sum([p == t for p, t in zip(pred.split(), target.split())]) / len(target.split())
                seq_acc = 1.0 if pred == target else 0.0
                token_accuracies.append(token_acc)
                sequence_accuracies.append(seq_acc)

    mean_token_acc = np.mean(token_accuracies)
    mean_seq_acc = np.mean(sequence_accuracies)
    print(f"Test Results for {model_suffix}: Token Accuracy = {mean_token_acc:.4f}, Sequence Accuracy = {mean_seq_acc:.4f}")

    return model, mean_token_acc, mean_seq_acc