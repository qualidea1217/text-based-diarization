import json
import os

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer

MAX_LENGTH = 512


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def get_test_results(model_code: str):
    tokenizer = RobertaTokenizer.from_pretrained(f"./{model_code}/tokenizer")
    with open(f"./{model_code}/{model_code}_test_2sp.json", 'r') as json_test:
        data_dict_test = json.load(json_test)
        texts_test = data_dict_test["text"]
        labels_test = data_dict_test["label"]
        print(labels_test.count(0))
        print(labels_test.count(1))
    dataset_test = Dataset.from_dict({"text": texts_test, "label": labels_test})

    def preprocess_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    dataset_test = dataset_test.map(preprocess_function, batched=True)
    for checkpoint in os.listdir(f"./{model_code}/results"):
        if "checkpoint" in checkpoint:
            model = RobertaForSequenceClassification.from_pretrained(os.path.join(f"./{model_code}/results", checkpoint))
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            results = trainer.evaluate(dataset_test)
            print(f"{model_code} {checkpoint} {results}")


if __name__ == "__main__":
    get_test_results("roberta-d7-u4-s1-21")
