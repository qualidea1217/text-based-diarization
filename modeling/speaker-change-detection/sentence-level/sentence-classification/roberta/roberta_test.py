import json
import random

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch


tokenizer = RobertaTokenizer.from_pretrained("roberta-large", cache_dir="./tokenizers")
model = RobertaForSequenceClassification.from_pretrained("./results/roberta-large-001/checkpoint-20001")

inputs = tokenizer(input_text, return_tensors='pt', max_length=512, padding="max_length", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label_idx = torch.argmax(logits, dim=1).item()
print(predicted_label_idx)