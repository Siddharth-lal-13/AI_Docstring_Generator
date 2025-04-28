# train.py
import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model

# Clear memory first
gc.collect()
torch.cuda.empty_cache()

# Configuration
MODEL_NAME = "t5-small"  # lighter model for 4GB GPU
SAVE_DIRECTORY = "saved_models/fine_tuned_model/"
NUM_EPOCHS = 5
BATCH_SIZE = 2

# Load dataset
dataset = load_dataset("mbpp")
print("Dataset loaded!")


# Preprocess dataset
def preprocess(example):
    code_snippet = example["code"]
    problem_desc = example["text"].lower().replace("write a function to ", "").replace("a ", "")
    return {
        "docstring_input": f"Write a docstring for:\n{code_snippet}",
        "docstring_target": f"Docstring: This function {problem_desc}",
        "comments_input": f"Add inline comments to:\n{code_snippet}",
        "comments_target": f"Comments: # This function {problem_desc}",
        "explanation_input": f"Explain the following function:\n{code_snippet}",
        "explanation_target": f"Explanation: This function {problem_desc} using an appropriate method."
    }


dataset = dataset.map(preprocess)

# Tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"], lora_dropout=0.05, bias="none",
                         task_type="SEQ_2_SEQ_LM")
model = get_peft_model(base_model, lora_config)
print("LoRA applied!")


# Tokenization functions
def tokenize_docstring(batch):
    inputs = tokenizer(batch["docstring_input"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(batch["docstring_target"], padding="max_length", truncation=True, max_length=128)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    return batch


def tokenize_comments(batch):
    inputs = tokenizer(batch["comments_input"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(batch["comments_target"], padding="max_length", truncation=True, max_length=128)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    return batch


def tokenize_explanation(batch):
    inputs = tokenizer(batch["explanation_input"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(batch["explanation_target"], padding="max_length", truncation=True, max_length=128)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    return batch


# Prepare datasets
train_docstring = dataset["train"].map(tokenize_docstring, batched=True)
train_comments = dataset["train"].map(tokenize_comments, batched=True)
train_explanation = dataset["train"].map(tokenize_explanation, batched=True)

train_docstring.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_comments.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_explanation.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./logs",
    learning_rate=5e-4,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=1,
    predict_with_generate=True,
    logging_dir="./logs",
    save_strategy="no",
    push_to_hub=False,
    fp16=True if torch.cuda.is_available() else False,
)

# Trainer for docstring
trainer_docstring = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_docstring,
)
print("Training docstring model...")
trainer_docstring.train()

# Trainer for comments
trainer_comments = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_comments,
)
print("Training comments model...")
trainer_comments.train()

# Trainer for explanation
trainer_explanation = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_explanation,
)
print("Training explanation model...")
trainer_explanation.train()

# Save final model
os.makedirs(SAVE_DIRECTORY, exist_ok=True)
model.save_pretrained(SAVE_DIRECTORY)
tokenizer.save_pretrained(SAVE_DIRECTORY)
print(f"✅ Fine-tuned model saved at {SAVE_DIRECTORY}")
