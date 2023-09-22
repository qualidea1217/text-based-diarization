import json
import random

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

MAX_LENGTH = 512
BATCH_SIZE = 24
EPOCHS = 3

MODEL_CODE = "roberta-d7-u8-s1-21"

# Load tokenizer and model
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir="./tokenizers")
tokenizer = RobertaTokenizer.from_pretrained(f"./{MODEL_CODE}/tokenizer")
model = RobertaForSequenceClassification.from_pretrained('roberta-large', cache_dir="./models", num_labels=2)
# If special tokens are added, remember to resize the model's embedding space
model.resize_token_embeddings(len(tokenizer))

# Load raw data
with open(f"./{MODEL_CODE}/{MODEL_CODE}_train.json", 'r') as json_train:
    data_dict_train = json.load(json_train)
    texts_train = data_dict_train["text"]
    labels_train = data_dict_train["label"]

with open(f"./{MODEL_CODE}/{MODEL_CODE}_val.json", 'r') as json_val:
    data_dict_val = json.load(json_val)
    texts_val = data_dict_val["text"]
    labels_val = data_dict_val["label"]

with open(f"./{MODEL_CODE}/{MODEL_CODE}_test.json", 'r') as json_test:
    data_dict_test = json.load(json_test)
    texts_test = data_dict_test["text"]
    labels_test = data_dict_test["label"]

# Create huggingface dataset
dataset_train = Dataset.from_dict({"text": texts_train, "label": labels_train})
dataset_val = Dataset.from_dict({"text": texts_val, "label": labels_val})
dataset_test = Dataset.from_dict({"text": texts_test, "label": labels_test})


def preprocess_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)


dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_val = dataset_val.map(preprocess_function, batched=True)
dataset_test = dataset_test.map(preprocess_function, batched=True)

# 3. Define Training Arguments and Initialize Trainer
training_args = TrainingArguments(
    output_dir=f'./{MODEL_CODE}/results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    optim="adamw_torch",
    save_strategy="epoch",
    evaluation_strategy="epoch"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    with open(f"./{MODEL_CODE}/eval_metrics.txt", 'a') as eval_log:
        eval_log.write(f"accuracy: {acc}, precision: {precision}, recall: {recall}, f1: {f1}\n")
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

