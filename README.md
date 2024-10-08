# LLM Fine-Tuning

This project demonstrates the fine-tuning of a Large Language Model (LLM) using the Hugging Face `Transformers` and `Datasets` libraries. The goal is to adapt a pre-trained model (MistralForCausalLM) to a specific task, leveraging efficient techniques such as low-bit quantization to optimize memory usage while maintaining model performance.

## Features
- **Model Architecture**: Utilizes the MistralForCausalLM, a decoder model suited for language modeling tasks.
- **Efficient Fine-tuning**: Applies 4-bit quantization via the `bitsandbytes` library for efficient computation on GPUs.
- **Dataset Handling**: Leverages the Hugging Face `datasets` library to load and preprocess text data, including tokenization using a pre-trained tokenizer.
- **Training Optimization**: Implements low-bit precision techniques to speed up training while maintaining performance, allowing for more memory-efficient operations on large-scale language models.

## Installation

Before running the code, install the necessary dependencies:
```bash
pip install torch datasets transformers bitsandbytes peft trl
```

Make sure that your system has a compatible NVIDIA CUDA environment for GPU support.

## Usage

### Step 1: Load the Dataset
The dataset used for fine-tuning contains text-label pairs, which are loaded using the Hugging Face `datasets` library. Example:
```python
from datasets import load_dataset
dataset = load_dataset('your_dataset_name')
```

### Step 2: Model Setup
Initialize the pre-trained model `MistralForCausalLM`:
```python
from transformers import MistralForCausalLM
model = MistralForCausalLM.from_pretrained('mistral_model')
```

### Step 3: Fine-Tuning
Set up the optimizer and fine-tune the model with low-bit precision:
```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-5)
# Continue with your fine-tuning loop
```

### Step 4: Evaluation
Once training is complete, evaluate the fine-tuned model on test data to measure its performance.

## Model Architecture

The model architecture is based on Mistral, featuring:
- 32 decoder layers with 4096-dimensional embeddings.
- 4-bit quantized attention and MLP layers for efficient computation.
- Layer normalization via `MistralRMSNorm`.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU
- Hugging Face Transformers library
- bitsandbytes for quantization
- PEFT for parameter-efficient fine-tuning

## Project Structure
- **Notebook**: `FineTuningLLM.ipynb` contains the end-to-end process of loading data, fine-tuning the model, and evaluating performance.
- **Datasets**: Text data used for fine-tuning.
- **Models**: Pre-trained LLMs from Hugging Face.

## License
This project is open-source under the MIT License.
