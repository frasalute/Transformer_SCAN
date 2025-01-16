import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AdamW, get_scheduler
from torch.cuda.amp import GradScaler, autocast
import numpy as np

def fine_tune_t5(train_dataset, test_dataset, hyperparams, model_suffix, random_seed, tokenizer):
    """Fine-tune the T5 model on the training dataset and evaluate on the test dataset."""
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Load model (Updated to T5-large)
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    model.to(hyperparams["device"])

    # Prepare DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])
    num_training_steps = len(train_dataloader) * hyperparams["epochs"]
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    model.train()
    for epoch in range(hyperparams["epochs"]):
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            # Forward pass with mixed precision
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / hyperparams["gradient_accumulation_steps"]

            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % hyperparams["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * hyperparams["gradient_accumulation_steps"]

        print(f"Epoch {epoch + 1}/{hyperparams['epochs']}, Loss: {total_loss / len(train_dataloader):.4f}")

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
