import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
from tqdm.auto import tqdm

# Step 1: Define Local Paths
print("Setting up the environment...")

# Get the directory where the script is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's location
train_file = os.path.join(base_path, "train1.json")
schema_file = os.path.join(base_path, "mia1.json")
output_dir = os.path.join(base_path, "results")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Step 2: Load the Dataset
print("Loading datasets...")

# Load training data
with open(train_file, 'r', encoding="utf-8") as f:
    train_data = json.load(f)['examples']

# Load schema data
with open(schema_file, 'r', encoding="utf-8") as f:
    schema_data = json.load(f)

# Step 3: Preprocess the Data
print("Preprocessing data...")

def format_schema(schema_data):
    schema_str = "DATABASE SCHEMA:\n"
    for table in schema_data:
        schema_str += f"Table: {table['table_name']}\nColumns:\n"
        for column in table['columns']:
            nullable = "NULL" if column['is_nullable'] == "YES" else "NOT NULL"
            schema_str += f"- {column['column_name']} ({column['data_type']}, {nullable})\n"
        
        if 'foreign_keys' in table and table['foreign_keys']:
            schema_str += "Foreign Keys:\n"
            for fk in table['foreign_keys']:
                schema_str += f"- {fk['column_name']} references {fk['referenced_table']}({fk['referenced_column']})\n"
        schema_str += "\n"
    return schema_str

schema_str = format_schema(schema_data)

def prepare_training_data(train_data, schema_str):
    sources = []
    targets = []
    for example in train_data:
        source = f"{schema_str}\nQuestion: {example['question']}"
        target = example['sql']
        sources.append(source)
        targets.append(target)
    return {"input": sources, "output": targets}

processed_data = prepare_training_data(train_data, schema_str)
dataset = Dataset.from_dict(processed_data)

# Split dataset into train and validation with stratification (if applicable)
# Using a larger validation set (20%) to better evaluate model performance
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Step 4: Load the Model and Tokenizer
print("Loading pre-trained model and tokenizer...")

model_name = "charanhu/text_to_sql_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Add dropout to prevent overfitting
# Modify model configuration to include dropout
model.config.dropout_rate = 0.2
model.config.attention_dropout = 0.1

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU if available
model.to(device)

# Step 5: Tokenization Setup
max_input_length = 1024  
max_output_length = 128  

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding="max_length", truncation=True
    )
    
    labels = tokenizer(
        targets, max_length=max_output_length, padding="max_length", truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    
    for i in range(len(labels["input_ids"])):
        model_inputs["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in labels["input_ids"][i]
        ]
    
    return model_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
)

# Step 6: Training Configuration with anti-overfitting measures
print("Setting up training configuration...")

# Increase batch size for OVHcloud
per_device_train_batch_size = 4
per_device_eval_batch_size = 4

# Early stopping implementation
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    # Lower learning rate
    learning_rate=2e-5,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    weight_decay=0.05,
    save_total_limit=3,
    # Reduce epochs
    num_train_epochs=4,
    # Enable early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    # Add warmup steps
    warmup_steps=200,
    # Add gradient clipping
    max_grad_norm=1.0,
    generation_max_length=max_output_length,
    report_to="none",
    label_smoothing_factor=0.1,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# Step 7: Trainer Setup with data shuffling enhancement
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train the Model
print("Starting fine-tuning process...")
trainer.train()

# Step 9: Save the Fine-Tuned Model
print("Saving the fine-tuned model...")
model.save_pretrained(os.path.join(base_path, "fine_tuned_text2sql_model"))
tokenizer.save_pretrained(os.path.join(base_path, "fine_tuned_text2sql_model"))

# Step 10: Evaluate the model on test set
print("Evaluating model on test set...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

print("Training and saving completed!")