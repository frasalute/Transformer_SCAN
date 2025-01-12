from torch.utils.data import Dataset
from transformers import T5Tokenizer

class SCANDataset(Dataset): 
    def __init__(self, file_path, tokenizer, max_len=128): 
        self.file_path = file_path 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._load_data()
    
    def _load_data(self): 
        data = [] 
        with open(self.file_path, 'r') as file: 
            for line in file: 
                line = line.strip() 
                if line.startswith("IN:") and "OUT:" in line: 
                    input_text = line.split("IN:")[1].split("OUT:")[0].strip() 
                    output_text = line.split("OUT:")[1].strip() 
                    data.append({"command": input_text, "action": output_text}) 
        return data
    
    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx): 
        src_text = self.data[idx]["command"]
        tgt_text = self.data[idx]["action"]
        
        # Tokenize source and target texts
        src_encoding = self.tokenizer(
            src_text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        tgt_encoding = self.tokenizer(
            tgt_text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": src_encoding["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": src_encoding["attention_mask"].squeeze(0),
            "labels": tgt_encoding["input_ids"].squeeze(0)
        }


if __name__ == "__main__": 
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = SCANDataset(
        "/work/Transformer_SCAN/data/simple_split/tasks_train_simple.txt", 
        tokenizer
    )
    
    # Example of how to access the dataset
    sample = dataset[0]
    print("Input IDs:", sample["input_ids"])
    print("Attention Mask:", sample["attention_mask"])
    print("Labels:", sample["labels"])
    
    # Decode for verification
    print("Decoded Input:", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
    print("Decoded Labels:", tokenizer.decode(sample["labels"], skip_special_tokens=True))