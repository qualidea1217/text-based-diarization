import json
import os
import random
import string

import numpy as np
import spacy
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer

MAX_LENGTH = 512

dir_dict = {"AMI audio": "/local/scratch/pwu54/Text-based SD Dataset/AMI/audio/",
            "AMI gt": "/local/scratch/pwu54/Text-based SD Dataset/AMI/transcript/",
            "AMI text": "/local/scratch/pwu54/Text-based SD Dataset/AMI/whisper_output/",
            "CallFriend audio": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/audio/",
            "CallFriend gt": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/transcript/",
            "CallFriend text": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/whisper_output/",
            "CallHome English audio": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/audio/",
            "CallHome English gt": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/transcript/",
            "CallHome English text": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/whisper_output/",
            "CHiME-5 audio": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/audio/",
            "CHiME-5 gt": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/transcript/",
            "CHiME-5 text": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/whisper_output/",
            "DailyTalk audio": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/audio/",
            "DailyTalk gt": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/transcript/",
            "DailyTalk text": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/whisper_output/",
            "ICSI audio": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/audio/",
            "ICSI gt": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/transcript/",
            "ICSI text": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/whisper_output/",
            "SBCSAE audio": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/audio/",
            "SBCSAE gt": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/transcript/",
            "SBCSAE text": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/whisper_output/"}


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


def get_train_val_test_filepath(gt_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of the ratios must equal 1.0")
    random.seed(seed)
    gt_filenames = os.listdir(gt_dir)
    random.shuffle(gt_filenames)
    total = len(gt_filenames)
    train_index = int(train_ratio * total)
    val_index = int((train_ratio + val_ratio) * total)
    train_filepaths = [os.path.join(gt_dir, filename) for filename in gt_filenames[:train_index]]
    val_filepaths = [os.path.join(gt_dir, filename) for filename in gt_filenames[train_index:val_index]]
    test_filepaths = [os.path.join(gt_dir, filename) for filename in gt_filenames[val_index:]]
    return train_filepaths, val_filepaths, test_filepaths


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


def get_conversation_w_speaker(conversation: list[str], speaker_label: list):
    return [[speaker_label[i], conversation[i]] for i in range(len(conversation))]


def preprocess_conversation(content, merge: bool = True, segment: bool = True):
    text_list = [utterance[1] for utterance in content if utterance[1] not in string.punctuation and utterance[1] not in string.whitespace]
    speaker_list = [utterance[0] for utterance in content if utterance[1] not in string.punctuation and utterance[1] not in string.whitespace]
    if merge:
        merged_texts = []
        merged_speakers = []
        j = 0
        while j < len(text_list):
            if (j < len(text_list) - 1) and (speaker_list[j] == speaker_list[j + 1]):
                merged_sentence = text_list[j]
                while (j < len(text_list) - 1) and (speaker_list[j] == speaker_list[j + 1]):
                    j += 1
                    merged_sentence += " " + text_list[j]
                merged_texts.append(merged_sentence)
                merged_speakers.append(speaker_list[j])
                j += 1
            else:
                merged_texts.append(text_list[j])
                merged_speakers.append(speaker_list[j])
                j += 1
        text_list = merged_texts
        speaker_list = merged_speakers
    if segment:
        spacy.require_gpu()  # if cupy is installed or spacy with gpu support is installed
        nlp = spacy.load("en_core_web_trf")
        sentences = []
        speaker_ids = []
        for j in range(len(text_list)):
            doc = nlp(text_list[j])
            for sent in doc.sents:
                sentences.append(sent.text)
                speaker_ids.append(speaker_list[j])
        text_list = sentences
        speaker_list = speaker_ids

    text_list_filter = [text_list[i] for i in range(len(text_list)) if text_list[i] not in string.punctuation and text_list[i] not in string.whitespace]
    speaker_list_filter = [speaker_list[i] for i in range(len(speaker_list)) if text_list[i] not in string.punctuation and text_list[i] not in string.whitespace]
    text_list, speaker_list = text_list_filter, speaker_list_filter
    return text_list, speaker_list


def get_conversation_metrics_exact(content: list, content_pred: list):
    """If both conversation have exact same words and 2 speaker only,
    both TDER and DF1 falls back to accuracy, which is 1 - WDER"""
    n_incorrect_utterance = 0
    n_total_utterance = 0
    n_incorrect_token = 0
    n_total_token = 0
    for i in range(len(content)):
        n_total_utterance += 1
        n_total_token += len(content[i][1].split())
        if content[i][0] != content_pred[i][0]:
            n_incorrect_utterance += 1
            n_incorrect_token += len(content[i][1].split())
    n_correct_token = n_total_token - n_incorrect_token
    n_correct_utterance = n_total_utterance - n_incorrect_utterance
    df1 = n_correct_token / n_total_token  # also accuracy on token
    tder = n_incorrect_token / n_total_token  # also wder
    acc_utterance = n_correct_utterance / n_total_utterance
    return df1, tder, acc_utterance


if __name__ == "__main__":
    val_filepath_all = []
    test_filepath_all = []
    for gt_dir in ["AMI gt", "CallFriend gt", "CallHome English gt", "CHiME-5 gt", "DailyTalk gt", "ICSI gt", "SBCSAE gt"]:
        gt_dir = dir_dict[gt_dir].replace("transcript", "whisper_align")
        train_filepaths, val_filepaths, test_filepaths = get_train_val_test_filepath(gt_dir)
        val_filepath_all.extend(val_filepaths)
        test_filepath_all.extend(test_filepaths)

    tder_list = []
    df1_list = []
    acc_u_list = []
    sen_num_list = []
    word_num_list = []
    for filepath in test_filepath_all:
        if os.path.splitext(filepath)[1] == ".json":
            with open(filepath, 'r') as json_in:
                content = json.load(json_in)  # fill in the content of conversation in this format
        conversation, speaker_label = preprocess_conversation(content)
        if len(set(speaker_label)) != 2:
            continue
        y_pred_list, speaker_label_pred = predict_dialogue(conversation, speaker_label, "roberta-d8-u4-s1-21")
        content_pred = get_conversation_w_speaker(conversation, speaker_label_pred)
        df1, tder, acc_u = get_conversation_metrics_exact(content, content_pred)
        sen_num = len(conversation)
        word_num = sum([len(sentence.split()) for sentence in conversation])
        print(f"filepath: {filepath}\nDF1: {df1}, TDER: {tder}, ACC_U: {acc_u}, sentence num: {sen_num} word num: {word_num}")
        tder_list.append(tder)
        df1_list.append(df1)
        acc_u_list.append(acc_u)
        sen_num_list.append(sen_num)
        word_num_list.append(word_num)
    print("\n==================================================================================")
    print(f"Val avg DF1: {sum(df1_list) / len(df1_list)}, Val avg TDER: {sum(tder_list) / len(tder_list)}, Val avg acc utterance: {sum(acc_u_list) / len(acc_u_list)}")
    total_weights = sum(sen_num_list)
    weighted_tder_sum = sum(t * w for t, w in zip(tder_list, sen_num_list))
    weighted_df1_sum = sum(d * w for d, w in zip(df1_list, sen_num_list))
    weighted_acc_u_sum = sum(a * w for a, w in zip(acc_u_list, sen_num_list))
    print(f"(Sentence-level Weighted) Val avg DF1: {weighted_df1_sum / total_weights}, Val avg TDER: {weighted_tder_sum / total_weights}, Val avg acc utterance: {weighted_acc_u_sum / total_weights}")
    total_weights = sum(word_num_list)
    weighted_tder_sum = sum(t * w for t, w in zip(tder_list, word_num_list))
    weighted_df1_sum = sum(d * w for d, w in zip(df1_list, word_num_list))
    weighted_acc_u_sum = sum(a * w for a, w in zip(acc_u_list, word_num_list))
    print(f"(Word-level Weighted) Val avg DF1: {weighted_df1_sum / total_weights}, Val avg TDER: {weighted_tder_sum / total_weights}, Val avg acc utterance: {weighted_acc_u_sum / total_weights}")
    print("==================================================================================\n")

    tder_list = []
    df1_list = []
    acc_u_list = []
    sen_num_list = []
    word_num_list = []
    for filepath in val_filepath_all:
        if os.path.splitext(filepath)[1] == ".json":
            with open(filepath, 'r') as json_in:
                content = json.load(json_in)  # fill in the content of conversation in this format
        conversation, speaker_label = preprocess_conversation(content)
        if len(set(speaker_label)) != 2:
            continue
        y_pred_list, speaker_label_pred = predict_dialogue(conversation, speaker_label, "roberta-d8-u4-s1-21")
        content_pred = get_conversation_w_speaker(conversation, speaker_label_pred)
        df1, tder, acc_u = get_conversation_metrics_exact(content, content_pred)
        sen_num = len(conversation)
        word_num = sum([len(sentence.split()) for sentence in conversation])
        print(
            f"filepath: {filepath}\nDF1: {df1}, TDER: {tder}, ACC_U: {acc_u}, sentence num: {sen_num} word num: {word_num}")
        tder_list.append(tder)
        df1_list.append(df1)
        acc_u_list.append(acc_u)
        sen_num_list.append(sen_num)
        word_num_list.append(word_num)
    print("\n==================================================================================")
    print(
        f"Test avg DF1: {sum(df1_list) / len(df1_list)}, Test avg TDER: {sum(tder_list) / len(tder_list)}, Test avg acc utterance: {sum(acc_u_list) / len(acc_u_list)}")
    total_weights = sum(sen_num_list)
    weighted_tder_sum = sum(t * w for t, w in zip(tder_list, sen_num_list))
    weighted_df1_sum = sum(d * w for d, w in zip(df1_list, sen_num_list))
    weighted_acc_u_sum = sum(a * w for a, w in zip(acc_u_list, sen_num_list))
    print(
        f"(Sentence-level Weighted) Test avg DF1: {weighted_df1_sum / total_weights}, Test avg TDER: {weighted_tder_sum / total_weights}, Test avg acc utterance: {weighted_acc_u_sum / total_weights}")
    total_weights = sum(word_num_list)
    weighted_tder_sum = sum(t * w for t, w in zip(tder_list, word_num_list))
    weighted_df1_sum = sum(d * w for d, w in zip(df1_list, word_num_list))
    weighted_acc_u_sum = sum(a * w for a, w in zip(acc_u_list, word_num_list))
    print(
        f"(Word-level Weighted) Test avg DF1: {weighted_df1_sum / total_weights}, Test avg TDER: {weighted_tder_sum / total_weights}, Test avg acc utterance: {weighted_acc_u_sum / total_weights}")
    print("==================================================================================\n")

    # randomly select first 2 conversation for observation
    # for gt_dir in ["AMI gt", "CallFriend gt", "CallHome English gt", "CHiME-5 gt", "DailyTalk gt", "ICSI gt", "SBCSAE gt"]:
    #     gt_dir = dir_dict[gt_dir].replace("transcript", "whisper_align")
    #     train_filepaths, val_filepaths, test_filepaths = get_train_val_test_filepath(gt_dir)
    #     count = 0
    #     for filepath in test_filepaths:
    #         if os.path.splitext(filepath)[1] == ".json":
    #             with open(filepath, 'r') as json_in:
    #                 content = json.load(json_in)  # fill in the content of conversation in this format
    #         conversation, speaker_label = preprocess_conversation(content)
    #         if len(set(speaker_label)) != 2:
    #             continue
    #         y_pred_list, speaker_label_pred = predict_dialogue(conversation, speaker_label, "roberta-d8-u4-s1-21")
    #         content_pred = get_conversation_w_speaker(conversation, speaker_label_pred)
    #         for i in range(len(content)):
    #             if i > 0:
    #                 print(f"Speaker change: {y_pred_list[i - 1]}")
    #             print(f"Correct: {speaker_label[i]} Predict: {speaker_label_pred[i]}")
    #             print(conversation[i])
    #         count += 1
    #         if count >= 2:
    #             break


