import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AdamW
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

def fine_tune_t5_with_lora(train_dataset, test_dataset, hyperparams, model_suffix, random_seed, tokenizer):

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.to(hyperparams["device"])

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,  
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"])
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    model.train()
    for epoch in range(hyperparams["epochs"]):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{hyperparams['epochs']}, Loss: {total_loss / len(train_dataloader)}")

    model.eval()
    token_accuracies = []
    sequence_accuracies = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(hyperparams["device"])
            attention_mask = batch["attention_mask"].to(hyperparams["device"])
            labels = batch["labels"].to(hyperparams["device"])

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, target in zip(predictions, targets):
                token_acc = sum(p == t for p, t in zip(pred.split(), target.split())) / len(target.split())
                seq_acc = 1.0 if pred == target else 0.0
                token_accuracies.append(token_acc)
                sequence_accuracies.append(seq_acc)

    mean_token_acc = np.mean(token_accuracies)
    mean_seq_acc = np.mean(sequence_accuracies)
    print(f"Test Results for {model_suffix}: Token Accuracy = {mean_token_acc:.4f}, Sequence Accuracy = {mean_seq_acc:.4f}")

    return model, mean_token_acc, mean_seq_acc
