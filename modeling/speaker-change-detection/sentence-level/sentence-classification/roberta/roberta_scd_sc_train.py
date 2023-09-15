import json
import random

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

MAX_LENGTH = 512
BATCH_SIZE = 24
EPOCHS = 5

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir="./tokenizers")
model = RobertaForSequenceClassification.from_pretrained('roberta-large', cache_dir="./models", num_labels=2)

# Load raw data
with open("/local/scratch/pwu54/Text-based SD Dataset/dataset7_roberta_scd_512_001.json", 'r') as json_in:
    data_dict = json.load(json_in)
    texts = data_dict["text"]
    labels = data_dict["label"]
    # randomly sample 20000 for verification
    # paired_data = list(zip(texts, labels))
    # random.seed(42)
    # random_data = random.sample(paired_data, 200000)
    # texts, labels = zip(*random_data)

# Down sample label 1 to make dataset balance
texts_label1 = [texts[i] for i in range(len(texts)) if labels[i] == 1]
texts_label0 = [texts[i] for i in range(len(texts)) if labels[i] == 0]
labels_label1 = [1] * len(texts_label1)
labels_label0 = [0] * len(texts_label0)
random.seed(42)
texts_label1_downsampled = random.sample(texts_label1, len(texts_label0))
labels_label1_downsampled = [1] * len(texts_label0)
new_texts = texts_label1_downsampled + texts_label0
new_labels = labels_label1_downsampled + labels_label0
combined = list(zip(new_texts, new_labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Create huggingface dataset
custom_dataset = Dataset.from_dict({"text": texts, "label": labels})
custom_dataset = custom_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
dataset_train = custom_dataset["train"]
dataset_test = custom_dataset["test"]


def preprocess_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)


dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_test = dataset_test.map(preprocess_function, batched=True)

# 3. Define Training Arguments and Initialize Trainer
training_args = TrainingArguments(
    output_dir='./results/roberta-large-001',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=1e-6,
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
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

