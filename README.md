# fine_tune_llm


# AI Medical Chatbot 3.0 – Fine-Tuning DeepSeek-R1

Welcome to the **AI Medical Chatbot 3.0** repo! This project sets up a state-of-the-art medical chatbot using DeepSeek-R1, fine-tuned for clinical reasoning with advanced chain-of-thought datasets. If you're preparing to discuss this in an interview, this README will help you explain all the steps and intentions clearly.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Environment Setup](#environment-setup)
- [Workflow Steps](#workflow-steps)
  - [Step 1: Import Libraries](#step-3-import-libraries)
  - [Step 2: Check HF Token](#step-4-check-hf-token)
  - [Step 3: Setup Pretrained DeepSeek-R1](#step-5-setup-pretrained-deepseek-r1)
  - [Step 4: System Prompt](#step-6-system-prompt)
  - [Step 5: Run Inference](#step-7-run-inference)
  - [Step 6: Fine-Tuning Setup](#step-8-fine-tuning-setup)
  - [Step 7: Apply LoRA Fine-Tuning](#step-9-apply-lora-fine-tuning)
  - [Step 8: Test After Fine-Tuning](#step-10-test-after-fine-tuning)
- [Interview Discussion Points](#interview-discussion-points)
- [References](#references)

## Overview

**AI Medical Chatbot 3.0** leverages a fine-tuned version of DeepSeek-R1, specialized for medical reasoning, diagnosis, and step-by-step clinical answers. The model is trained using advanced prompt engineering and efficient LoRA methods for practical deployment, focusing on interpretability and real-world clinical applicability.

## Key Features

- Uses **DeepSeek-R1-Distill-Llama-8B** as the backbone.
- End-to-end fine-tuning pipeline based on real clinical case datasets (CoT).
- Efficient LoRA fine-tuning for resource savings.
- Easily extendable for various clinical NLU/NLP tasks.
- Integrated with Weights & Biases (wandb) for experiment tracking.

## Environment Setup

Ensure your system (preferably Google Colab with GPU) is prepared:

```bash
pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

pip install trl==0.14.0 peft==0.14.0 xformers==0.0.28.post3
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

## Workflow Steps

### Step 1: Import Libraries

Import all necessary modules for model loading, dataset handling, training, and tracking.

```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset
import wandb
```

### Step 2: Check HF Token

Authenticate using your Hugging Face token to ensure you can download models and datasets.

```python
from google.colab import userdata
hf_token = userdata.get('HUGGINGFACE_API_KEY')
login(hf_token)
```

Optional: Check CUDA GPU support.

```python
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### Step 3: Setup Pretrained DeepSeek-R1

Download and initialize the DeepSeek-R1 model in 4-bit quantized mode for memory efficiency.

```python
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    token=hf_token
)
```

### Step 4: System Prompt

Establish a multi-step reasoning prompt for all queries.

```python
prompt_style = """
Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

### Task:
You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge.

### Query:
{}

### Answer:
{}
"""
```

### Step 5: Run Inference

Prepare a test question and generate a response with the base model.

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Answer:")[1])
```

### Step 6: Fine-Tuning Setup

Load the advanced medical reasoning dataset (few-shot, chain-of-thought style).

```python
medical_dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[:500]", trust_remote_code=True)
```

Format prompts and process for fine-tuning.

```python
def preprocess_input_data(examples):
    # Prepare prompt using CoT
    ...
finetune_dataset = medical_dataset.map(preprocess_input_data, batched=True)
```

### Step 7: Apply LoRA Fine-Tuning

Add LoRA adaptors for efficient model update.

```python
model_lora = FastLanguageModel.get_peft_model(...)
trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    train_dataset=finetune_dataset,
    ...
)
```

Monitor with wandb:

```python
wandb.login(key=wnb_token)
run = wandb.init(project='Fine-tune-DeepSeek-R1-on-Medical-CoT-Dataset_svk', job_type="training", anonymous="allow")
trainer_stats = trainer.train()
wandb.finish()
```

### Step 8: Test After Fine-Tuning

Switch to inference mode and query the fine-tuned model with new clinical scenarios for evaluation.

```python
FastLanguageModel.for_inference(model_lora)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model_lora.generate(...)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Answer:")[1])
```

## Interview Discussion Points

- **Prompt engineering**: Explain how careful construction guides the model toward step-by-step clinical reasoning.
- **LoRA fine-tuning**: Describe why LoRA is used — to efficiently update only a subset of parameters, saving compute/memory.
- **Data selection**: The use of chain-of-thought medical CoT datasets ensures the bot learns to reason, not just answer.
- **Inference and evaluation**: How you evaluate effectiveness before and after fine-tuning.
- **Experiment tracking**: Emphasize the importance of wandb for reproducibility and tracking training metrics.

## References

- DeepSeek-R1 official HF repo
- FreedomIntelligence/medical-o1-reasoning-SFT dataset
- Unsloth documentation

**Pro Tip for Interviews:**  
Focus not just on code, but on *why* each step is needed (prompt design, data, LoRA), and what value it adds for the end-user—especially in medical safety and model transparency.