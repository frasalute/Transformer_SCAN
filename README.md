# Transformer Implementation and Experiments

## Introduction
This project focuses on the reimplementation of the Transformer architecture, based on the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and subsequent experiments on the SCAN dataset, inspired by the paper [Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks](https://arxiv.org/abs/1811.12884). The goal was to reproduce key results and explore the effectiveness of pretrained Transformer T5 on the SCAN dataset.

## Project Structure
The project includes implementations of various Transformer models, dataset loaders, and experiments using both standard and advanced methods like Adafactor optimization and LoRA. Below is an overview of the key files:

### Dataset Handling
- **`dataset.py`**: Loads and preprocesses the SCAN dataset.
- **`t5_dataset.py`**: Prepares the dataset for use with T5 models.

### Transformer Models
- **`transformer.py`**: Core implementation of the Transformer architecture.
- **`adafactort5_model.py`**: Adafactor-based T5 model implementation.
- **`t5_model.py`**: Small T5 model implementation.
- **`t5_model_large.py`**: Implementation of the Large T5 model.
- **`lora_model.py`**: LoRA (Low-Rank Adaptation) model implementation.

### Experiments
#### Adafactor Experiments Small T5 Transformer
- **`adafactort5_exp1.py`**: Conducts Experiment 1 using Adafactor optimization.
- **`adafactort5_exp2.py`**: Conducts Experiment 2 using Adafactor optimization.

#### Greedy Decoding Experiments on Attention is all you need Transformer
- **`greedy_exp_1.py`**: Experiment 1 with Greedy decoding.
- **`greedy_exp_2.py`**: Experiment 2 with Greedy decoding.
- **`greedy_exp_3.py`**: Experiment 3 with Greedy decoding.
- **`greedy_exp_3best.py`**: Best-case implementation of Experiment 3 with Greedy decoding.

#### T5 Experiments
- **`t5_exp_1.py`**: Conducts Experiment 1 using Small T5.
- **`t5_exp_2.py`**: Conducts Experiment 2 using Small T5.
- **`t5large_exp.py`**: Runs experiments with the Large T5 model.

#### LoRA Experiments
- **`lora_exp.py`**: Conducts experiments using LoRA.

### Training on Attention is all you need Transformer
- **`train_greedysearch.py`**: Implements training using Greedy Search decoding.

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**
   Ensure you have Python 3.8+ installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Download the SCAN dataset and ensure it is correctly placed or referenced.

## Running Experiments
### Adafactor Experiments
To run Experiment 1 with Adafactor optimization:
```bash
python adafactort5_exp1.py
```

### Greedy Decoding Experiments
For Experiment 3 with Greedy decoding:
```bash
python greedy_exp_3.py
```

### T5 Experiments
Run Experiment 2 with the T5 model:
```bash
python t5_exp_2.py
```

### LoRA Experiments
To run LoRA experiments:
```bash
python lora_exp.py
```

## Results
The experiments yielded the following key findings:
1. **Adafactor Optimization**: Best results obtained.
2. **Small T5 Performance**: Effective on smaller datasets but limited generalization to longer sequences.
3. **LoRA**: Showed potential for scalability but faced computational constraints with current resources.

Detailed results are available in the project report.

## Limitations
- Computational constraints, suggested to run with at least 3GPU. 


