یک فایل `README.md` جامع، ساختاریافته و استاندارد بر اساس نوت‌بوک، متغیرها، هایپرپارامترها و جزئیات ارزیابی شما آماده شده است. همان‌طور که خواستید، نکته مربوط به آموزش روی زیرمجموعه دیتاست به دلیل محدودیت سخت‌افزاری و پتانسیل بهبود با اسکیل کردن ترین به‌طور کامل در آن گنجانده شده است.

---

```markdown
# 💬 Instruction Fine-Tuning Gemma on Databricks Dolly-15k with LoRA & Unsloth

A lightweight, parameter-efficient instruction-tuning pipeline for **Gemma (2B)** on curated subsets of the **Databricks Dolly-15k** dataset, accelerated using **Unsloth (4-bit QLoRA)** and **TRL's SFTTrainer**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/%F0%9F%A6%A5%20Unsloth-4--bit%20Patching-green.svg)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)

```

---

## 📌 Overview

This project demonstrates how to turn a base language model (**Gemma 2B**) into an instruction-following assistant using supervised fine-tuning (SFT).

By leveraging **Unsloth 4-bit quantization** and **Low-Rank Adaptation (LoRA)**, the entire training pipeline fits comfortably within commodity consumer hardware (e.g., a free-tier Tesla T4 with 16GB VRAM), achieving rapid convergence and steady loss reduction.

### Key Highlights:

* **Base LLM**: `unsloth/gemma-4-E2B` loaded in 4-bit precision.
* **PEFT / LoRA**: Rank $r = 8$, $\alpha = 8$ applied to both attention and MLP projection layers.
* **Completion Masking**: Employed `train_on_responses_only` to mask user instructions (`<|turn>user\n`) and backpropagate gradients strictly on model completions (`<|turn>model\n`).
* **Structured Evaluation**: Evaluated completions on held-out test instructions and logged predictions to `test_predictions.json`.

---

## ⚠️ Hardware Constraints & Scaling Potential

> **Note on Scope**: Due to compute resource and GPU memory limitations (Google Colab T4 environment), this run was intentionally executed on a **balanced sub-sample** of the dataset for **120 training steps** with an effective batch size of 16.
> **Scaling Up**:
> * **Full Dataset**: Scaling training to the complete 15,000 instructions in Dolly-15k will provide broader generalization across more categories.
> * **Epochs & Schedule**: Increasing the training length to 2–3 full epochs with extended warmup steps and a higher LoRA rank ($r=16$ or $r=32$) is expected to significantly improve factual accuracy, formatting adherence, and complex multi-turn reasoning.
> 
> 

---

## 📊 Dataset & Preprocessing

We use the [databricks/databricks-dolly-15k](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fdatabricks%2Fdatabricks-dolly-15k) instruction-following dataset.

1. **Category Filtering**: Filtered down to 4 representative task categories:
* `open_qa`
* `general_qa`
* `classification`
* `closed_qa`


2. **Balancing**: Balanced the target categories against `closed_qa` (1,773 examples per class), resulting in **5,319 total processed samples**.
3. **Data Splitting**:
* **Train**: 85% (~4,521 samples)
* **Validation**: 5% (~267 samples)
* **Test**: 10% (~531 samples)


4. **Chat Template Format**: Applied standard Gemma conversational tokens:
```text
<|turn>user
{Instruction}

Input:
{Input (if present)}<|turn>model
{Output}

```



---

## ⚙️ Hyperparameters & Training Setup

| Parameter | Configuration | Details |
| --- | --- | --- |
| **Base Model** | `unsloth/gemma-4-E2B` | 2B Parameter Model |
| **Quantization** | 4-bit (bitsandbytes) | QLoRA for low VRAM consumption |
| **Max Sequence Length** | 1024 tokens | Truncation safe |
| **LoRA Config** | $r = 8, \alpha = 8$, Dropout = 0 | Attention + MLP modules |
| **Batch Size** | 2 per device | Gradient Accumulation Steps = 8 (Effective batch = 16) |
| **Optimizer** | `adamw_8bit` | VRAM-saving 8-bit optimizer |
| **Learning Rate** | `2e-4` | Cosine decay schedule with 5 warmup steps |
| **Training Steps** | 120 steps | Checkpoints and validation every 15 steps |
| **Best Model Metric** | `eval_loss = 0.8792` | Restored lowest loss checkpoint |

---

## 📈 Training Progression

The model demonstrated consistent convergence, with the validation loss dropping steadily from `0.958` down to `0.879`.

| Step | Training Loss | Validation Loss | Notes |
| --- | --- | --- | --- |
| 15 | 1.4579 | 0.9587 | Initial convergence |
| 30 | 0.9878 | 0.9082 |  |
| 45 | 1.4451 | 0.8970 |  |
| 60 | 1.4328 | 0.8883 |  |
| 75 | 1.3434 | 0.8868 |  |
| 90 | 1.2155 | 0.8818 |  |
| 105 | 1.2605 | 0.8796 |  |
| **120** | **1.6596** | **0.8792** | **Best Validation Loss Checkpoint** |

---

## 🧪 Evaluation & Test Predictions

The model was tested using greedy decoding with light repetition penalty:

* `temperature`: `0.1`
* `repetition_penalty`: `1.15`
* `max_new_tokens`: `256`

Generated responses are logged in JSON format under `test_predictions.json`:

```json
[
  {
    "instruction": "Which is a species of fish? Tope or Rope",
    "input": "",
    "ground_truth": "Tope",
    "prediction": "Tope is a species of fish."
  }
]

```

---

## 🚀 Quickstart & Inference

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)/.git
cd 

# Install pinned dependencies
pip install -r requirements.txt

```

### 2. Run Inference with Fine-Tuned LoRA

```python
import torch
from unsloth import FastModel, FastLanguageModel

# Load the fine-tuned LoRA checkpoint
model_path = "gemma_4_lora"
model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Define your custom prompt
instruction = "Explain why the sky looks blue to a five-year-old."
messages = [
    {"role": "user", "content": [{"type": "text", "text": instruction}]}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Generate response
prompt_len = inputs.shape[1]
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=0.1,
        repetition_penalty=1.15
    )

response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
print(f"Assistant: {response}")

```

---

## 📂 Repository Structure

```text
├── Gemma_Dolly_SFT.ipynb          # Complete training & inference notebook
├── test_predictions.json          # Sample inference results on test split
├── requirements.txt               # Dependencies
└── README.md                      # Documentation

```

---

## 📜 Acknowledgements

* [Databricks](https://github.com/databrickslabs/dolly) for open-sourcing the Dolly-15k dataset.
* [Google Gemma Team](https://huggingface.co/google) for the base weights.
* [Unsloth AI](https://github.com/unslothai/unsloth) for low-overhead fine-tuning and fast patching.

---
