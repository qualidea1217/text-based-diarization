import json
import os

import spacy
from tqdm import trange, tqdm

import numpy as np
import torch
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
    checkpoint_dir = f"./{model_code}/results/2sp"
    for checkpoint in os.listdir(checkpoint_dir):
        if "checkpoint" in checkpoint:
            model = RobertaForSequenceClassification.from_pretrained(os.path.join(checkpoint_dir, checkpoint))
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            results = trainer.evaluate(dataset_test)
            print(f"{model_code} {checkpoint} {results}")


def predict_single_input(tokenizer, model, input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        proba = torch.nn.functional.softmax(logits, dim=1)
        y_pred = torch.argmax(proba, dim=1).item()
    return y_pred


def recreate_speaker_label_2sp(conversation: list[str], y_pred_list: list, speaker_label: None | list):
    if len(conversation) - 1 != len(y_pred_list):
        raise ValueError("Length of y_pred_list should be one less than the length of conversation.")
    speaker_label_int = [0]
    for change in y_pred_list:
        current_speaker = (speaker_label_int[-1] + change) % 2
        speaker_label_int.append(current_speaker)
    if speaker_label:
        seen = {}
        occurrence2label = []
        for label in speaker_label:
            if label not in seen:
                seen[label] = len(seen)
                occurrence2label.append(label)
            if len(occurrence2label) >= 2:
                break
        speaker_label_pred = [occurrence2label[i] for i in speaker_label_int]
    else:
        speaker_label_pred = speaker_label_int
    return speaker_label_pred


def predict_dialogue(conversation: list[str], speaker_label: list, model_code: str):
    SEPARATION = " </s> </s> "
    MODEL_CODE = model_code

    MODEL_CODE_SEGMENT = MODEL_CODE.split('-')
    HISTORY_UTTERANCE_NUM = int([s[1:] for s in MODEL_CODE_SEGMENT if s.startswith("u")][0])
    FUTURE_SENTENCE_NUM = int([s[1:] for s in MODEL_CODE_SEGMENT if s.startswith("s")][0])
    DATASET_NUM = int([s[1:] for s in MODEL_CODE_SEGMENT if s.startswith("d")][0])
    HISTORY_UTTERANCE_START = [" "]
    if int(MODEL_CODE_SEGMENT[-1][0]) == 0:
        HISTORY_UTTERANCE_START = [" "]
    elif int(MODEL_CODE_SEGMENT[-1][0]) == 1:
        HISTORY_UTTERANCE_START = [" </u> "]
    elif int(MODEL_CODE_SEGMENT[-1][0]) == 2:
        HISTORY_UTTERANCE_START = [f" </u{i}> " for i in range(1, HISTORY_UTTERANCE_NUM + 1)]
    FUTURE_START = " "
    if int(MODEL_CODE_SEGMENT[-1][1]) == 0:
        FUTURE_START = " "
    elif int(MODEL_CODE_SEGMENT[-1][1]) == 1:
        FUTURE_START = " </n> "

    def get_history_sequence(history: list, history_speaker: list) -> str:
        index = 1
        history_sequence = HISTORY_UTTERANCE_START[0] + history[0]
        for i in range(1, len(history)):
            if history_speaker[i] == history_speaker[i - 1]:
                history_sequence += " " + history[i]
            else:
                if len(HISTORY_UTTERANCE_START) > 1:
                    history_sequence += HISTORY_UTTERANCE_START[index] + history[i]
                    index += 1
                else:
                    history_sequence += HISTORY_UTTERANCE_START[0] + history[i]
        return history_sequence

    # load tokenizer and model
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir="./tokenizers")
    tokenizer = RobertaTokenizer.from_pretrained(f"./{MODEL_CODE}/tokenizer")
    # model = RobertaForSequenceClassification.from_pretrained('roberta-large', cache_dir="./models", num_labels=2)
    model = RobertaForSequenceClassification.from_pretrained(f'./{MODEL_CODE}/results/2sp/checkpoint-22860', num_labels=2)
    model = model.to("cuda")

    # preprocess the conversation
    input_context_list = []
    for i in range(1, len(conversation)):
        target = conversation[i]
        history = [conversation[i - 1]]
        history_speaker = [speaker_label[i - 1]]
        if FUTURE_SENTENCE_NUM == 0:
            future = ""
        else:
            future = FUTURE_START + " ".join(conversation[i + 1:i + 1 + FUTURE_SENTENCE_NUM])
        speaker_change = 0
        for j in range(i - 2, -1, -1):
            if HISTORY_UTTERANCE_NUM == 0:  # if equals to 0, not consider utterance and fill until max length
                history.insert(0, conversation[j])
                history_speaker.insert(0, speaker_label[j])
                input_context = get_history_sequence(history, history_speaker) + SEPARATION + target + future
                if len(tokenizer.encode(input_context)) > MAX_LENGTH:
                    history.pop(0)
                    history_speaker.pop(0)
                    break
            else:
                # Detect number of utterance in the history
                if speaker_label[j] != speaker_label[j + 1]:
                    speaker_change += 1
                if speaker_change >= HISTORY_UTTERANCE_NUM:  # Specify the number of utterance in history
                    break
                # If utterance is not fill out, continue to add sentence to history
                history.insert(0, conversation[j])
                history_speaker.insert(0, speaker_label[j])
        # Set label according to speaker change or not (0 for unchange, 1 for change)
        input_context = get_history_sequence(history, history_speaker) + SEPARATION + target + future
        input_context_list.append(input_context)

    # do prediction
    y_pred_list = [predict_single_input(tokenizer, model, input_context) for input_context in tqdm(input_context_list)]
    speaker_label_pred = recreate_speaker_label_2sp(conversation, y_pred_list, speaker_label)
    return y_pred_list, speaker_label_pred


def get_conversation_pred(conversation: list[str], speaker_label: list):
    return [[speaker_label[i], conversation[i]] for i in range(len(conversation))]


if __name__ == "__main__":
    conversation = [
        "Did you catch the latest episode of 'Galactic Explorers'?",
        "I did, and I was blown away by the plot twists!",
        "Right? I never saw that coming.",
        "Especially the part with the alien council.",
        "I was on the edge of my seat the entire time.",
        "And that cliffhanger at the end...",
        "I can't wait for next week's episode."
    ]

    speaker_label = [
        "Speaker_A",
        "Speaker_B",
        "Speaker_A",
        "Speaker_B",
        "Speaker_B",
        "Speaker_A",
        "Speaker_A"
    ]

    y_pred_list, speaker_label_pred = predict_dialogue(conversation, speaker_label, "roberta-d8-u4-s1-21")
    print(y_pred_list)
    print(speaker_label_pred)
