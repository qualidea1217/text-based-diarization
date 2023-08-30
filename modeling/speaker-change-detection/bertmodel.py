import json

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 5
HISTORY_SEPARATION = " [SEP] "
HISTORY_LAST_SEPARATION = " [SEP] "
TARGET_SEPARATION = " [SEP] "

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', cache_dir="./tokenizers")
model = BertForSequenceClassification.from_pretrained('bert-large-cased', cache_dir="./models", num_labels=2)

# Load raw data
with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
    data_dict = json.load(json_in)
    conversations = data_dict["text_list"]
    speaker_labels = data_dict["speaker_list"]

texts = []
labels = []

# Convert raw data to formatted data with length limitation
for conversation, speaker_label in zip(conversations, speaker_labels):
    for i in range(1, len(conversation)):
        input_context = conversation[i - 1] + TARGET_SEPARATION + conversation[i]
        if len(tokenizer.tokenize(input_context)) > MAX_LENGTH:
            continue
        for j in range(i - 2, -1, -1):
            if j == i - 2:
                input_context_temp = conversation[j] + HISTORY_LAST_SEPARATION + input_context
            else:
                input_context_temp = conversation[j] + HISTORY_SEPARATION + input_context
            if len(tokenizer.tokenize(input_context_temp)) > MAX_LENGTH:
                break
            else:
                input_context = input_context_temp
        label = 1 if speaker_label[i - 1] == speaker_label[i] else 0
        texts.append(input_context)
        labels.append(label)

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
    output_dir='./results/bert-large-cased',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
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

