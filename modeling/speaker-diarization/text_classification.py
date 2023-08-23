import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForTokenClassification
from datasets import Dataset
import evaluate

import torch

from preprocessing.pure_text_transcript import get_interview_dataset_for_token_classification


def bert_with_interview():
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", cache_dir="./tokenizers")
    dataset, max_speaker = get_interview_dataset_for_token_classification(tokenizer)
    model = AutoModelForTokenClassification.from_pretrained('bert-large-uncased', num_labels=max_speaker, cache_dir="./models")

    def encode_function(example):
        # Encoding the text
        encoding = tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)
        encoding['labels'] = [example['labels'][i] if i < len(example['labels']) else -100 for i in range(512)]
        return encoding

    encoded_dataset = dataset.map(encode_function, batched=True)
    split_dataset = encoded_dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Flatten the outputs
        labels_flat = labels.flatten()
        preds_flat = preds.flatten()

        # Accuracy
        acc = accuracy_score(labels_flat, labels_flat != -100)  # Ignore -100 labels

        # WDER
        incorrect_predictions = sum([(true_label != pred_label and true_label != -100) for true_label, pred_label in
                                     zip(labels_flat, preds_flat)])
        total_valid_tokens = sum(labels_flat != -100)
        wder = incorrect_predictions / total_valid_tokens

        return {'accuracy': acc, 'wder': wder}

    training_args = TrainingArguments(
        output_dir="./bert",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate(val_dataset)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", cache_dir="./tokenizers")
    dataset, max_speaker = get_interview_dataset_for_token_classification(tokenizer)
    print(dataset)
    print(dataset[0])
    print(dataset[0]["text"])
    print(dataset[0]["labels"])
    print(dataset[0]["attention_mask"])
    print(max_speaker)
    # bert_with_interview()
