import json

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

MAX_LENGTH = 512
BATCH_SIZE = 128
EPOCHS = 5

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("./results/roberta-large-001/checkpoint-20001")
model = RobertaForSequenceClassification.from_pretrained('roberta-large', cache_dir="./models", num_labels=2)
model.to("cuda")

with open("/local/scratch/pwu54/Text-based SD Dataset/dataset7_roberta_scd_512_001.json", 'r') as json_in:
    data_dict = json.load(json_in)
    texts = data_dict["text"]
    labels = data_dict["label"]

# Create huggingface dataset
custom_dataset = Dataset.from_dict({"text": texts, "label": labels})


def preprocess_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)


custom_dataset = custom_dataset.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


training_args = TrainingArguments(
    output_dir="./results/eval",
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    eval_dataset=custom_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate(eval_dataset=custom_dataset)
print(metrics)