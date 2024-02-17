import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

def preprocess_function(sample, padding="max_length"):
    model_inputs = tokenizer(sample["article"], max_length=256, padding=padding, truncation=True)
    labels = tokenizer(sample["highlights"], max_length=256, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
valid_tokenized_dataset = valid_data.map(preprocess_function, batched=True, remove_columns=valid_data.column_names)
