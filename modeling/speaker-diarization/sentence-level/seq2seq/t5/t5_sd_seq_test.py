import copy
import json
import os
import random
import string

import numpy as np
import spacy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


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

# Hyper parameters
CHANGE_POINT = " <change> "
MAX_SPEAKER = float("inf")


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []
    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict)
        output_list.append(unique_dict[item])
    return output_list


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


def get_conversation_w_speaker(conversation: list[str], speaker_label: list):
    return [[speaker_label[i], conversation[i]] for i in range(len(conversation))]


def get_global_label(speaker_label_pred: list, speaker_num: int | None = None) -> list:
    speaker_label_pred_global = copy.deepcopy(speaker_label_pred)
    for i in range(len(speaker_label_pred_global) - 1):
        if speaker_label_pred_global[i][0][0] == 0 and speaker_label_pred_global[i + 1][0][0] == 0:
            sentence_range, label = next((e for e in reversed(speaker_label_pred_global) if e[0][0] == 0))
            sentence_range_next, label_next = speaker_label_pred_global[i]
        else:
            sentence_range, label = speaker_label_pred_global[i]
            sentence_range_next, label_next = speaker_label_pred_global[i + 1]
        intersection = (max(sentence_range[0], sentence_range_next[0]), min(sentence_range[1], sentence_range_next[1]))
        label_index_range = (intersection[0] - sentence_range[0], intersection[1] - sentence_range[0])
        label_next_index_range = (intersection[0] - sentence_range_next[0], intersection[1] - sentence_range_next[0])
        label_map = find_label_mapping_int(label[label_index_range[0]:label_index_range[1]],
                                           label_next[label_next_index_range[0]:label_next_index_range[1]])
        label_next_global = [label_map.get(ln, '*') for ln in label_next]

        # Handle new speaker that is not in the mapping (new speaker in the end)
        for j in range(len(label_next)):
            if label_next_global[j] == '*':
                if speaker_num is None:
                    label_next_global[j] = max([e for e in label_next_global if e != '*']) + 1
                else:
                    label_next_global[j] = (max([e for e in label_next_global if e != '*']) + 1) % speaker_num

        if speaker_label_pred_global[i][0][0] == 0 and speaker_label_pred_global[i + 1][0][0] == 0:
            speaker_label_pred_global[i] = (sentence_range_next, label_next_global)
        else:
            speaker_label_pred_global[i + 1] = (sentence_range_next, label_next_global)
    return speaker_label_pred_global


def find_label_mapping_int(ref_labels: list[int], new_labels: list[int], sentence_lengths: list[int] = None) -> dict:
    # Use the previously defined function to find the label mapping for the new lists
    # Adjusting the function to work with integer labels within the execution cell for direct execution
    # potential improvement: add length of sentence as a weight
    if len(ref_labels) != len(new_labels):
        raise ValueError("Reference and new labels must have the same length.")
    if sentence_lengths is None:
        sentence_lengths = [1] * len(new_labels)  # default weight as 1 (no weight)
    unique_ref_labels = sorted(set(ref_labels))
    unique_new_labels = sorted(set(new_labels))
    ref_label_to_index = {label: index for index, label in enumerate(unique_ref_labels)}
    new_label_to_index = {label: index for index, label in enumerate(unique_new_labels)}
    confusion_matrix = np.zeros((len(unique_ref_labels), len(unique_new_labels)), dtype=int)
    for ref_label, new_label, sentence_length in zip(ref_labels, new_labels, sentence_lengths):
        ref_index = ref_label_to_index[ref_label]
        new_index = new_label_to_index[new_label]
        confusion_matrix[ref_index, new_index] += sentence_length
    cost_matrix = confusion_matrix.max() - confusion_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {unique_new_labels[new]: unique_ref_labels[ref] for ref, new in zip(row_ind, col_ind)}
    return mapping


def aggregate(speaker_label_pred: list, conversation_length: int) -> list:
    aggregation_log = [{} for _ in range(conversation_length)]
    for sentence_range, label in speaker_label_pred:
        for i in range(sentence_range[0], sentence_range[1]):
            sl = label[i - sentence_range[0]]
            aggregation_log[i][sl] = aggregation_log[i].get(sl, 0) + 1  # increment by 1
    aggregation_result = [max(d, key=lambda k: d[k]) for d in aggregation_log]
    return aggregation_result


def get_conversation_metrics_exact(content: list, content_pred: list):
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
    wder = n_incorrect_token / n_total_token  # also tder
    acc_utterance = n_correct_utterance / n_total_utterance
    return df1, wder, acc_utterance


def predict_single_input(model, tokenizer, input_text) -> list[int]:
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return [int(c) for c in decoded_output if c in string.digits]


def predict_conversation(model, tokenizer, conversation: list[str], speaker_label: list,
                         min_sentence_num: int = 2, max_sentence_num: int | float = float("inf")):
    """
    Using sliding window to predict each point of change for multiple times and do majority vote
    max_sentence_num must be even number so that for each
    """
    speaker_label_pred = []
    for i in range(min_sentence_num, len(conversation) + max_sentence_num - min_sentence_num + 1):
        begin = i - max_sentence_num if i - max_sentence_num >= 0 else 0
        end = i if i <= len(conversation) else len(conversation)
        single_input = CHANGE_POINT.join([sentence for sentence in conversation[begin:end]])
        single_output = predict_single_input(model, tokenizer, single_input)
        speaker_label_pred.append(((begin, end), single_output))
    speaker_label_pred_global = get_global_label(speaker_label_pred, len(set(speaker_label)))
    aggregation_result = aggregate(speaker_label_pred_global, len(conversation))
    for i in range(len(speaker_label_pred)):
        label_range, label_pred = speaker_label_pred[i][0], speaker_label_pred[i][1]
        label_ref = speaker_label[label_range[0]:label_range[1]]
        label_pred_global = speaker_label_pred_global[i][1]
        label_agg = aggregation_result[label_range[0]:label_range[1]]
        print(f"label range: {label_range}")
        print(f"label_ref: {label_ref}")
        print(f"label_pred: {label_pred}")
        print(f"label_pred_global: {label_pred_global}")
        print(f"label_agg: {label_agg}")
    return aggregation_result


def evaluate_conversation(model, tokenizer, conversation: list[str], speaker_label: list,
                          min_sentence_num: int = 2, max_sentence_num: int | float = float("inf")):
    speaker_label = speaker_to_ints(speaker_label)
    speaker_label_pred = predict_conversation(model, tokenizer, conversation, speaker_label, min_sentence_num, max_sentence_num)
    final_mapping = find_label_mapping_int(speaker_label, speaker_label_pred, [len(s.split()) for s in conversation])
    print(final_mapping)
    speaker_label_pred = [final_mapping[l] for l in speaker_label_pred]
    print(f"speaker_label_pred: {speaker_label_pred}")
    print(f"speaker_label: {speaker_label}")
    content = get_conversation_w_speaker(conversation, speaker_label)
    content_pred = get_conversation_w_speaker(conversation, speaker_label_pred)
    df1, tder, acc_utterance = get_conversation_metrics_exact(content, content_pred)
    return df1, tder, acc_utterance


def evaluate_checkpoint(checkpoint: str, min_sentence_num: int = 2, max_sentence_num: int | float = float("inf")):
    # load data
    val_filepath_all = []
    test_filepath_all = []
    for gt_dir in ["AMI gt", "CallFriend gt", "CallHome English gt", "CHiME-5 gt", "DailyTalk gt", "ICSI gt",
                   "SBCSAE gt"]:
        gt_dir = dir_dict[gt_dir].replace("transcript", "whisper_align")
        train_filepaths, val_filepaths, test_filepaths = get_train_val_test_filepath(gt_dir)
        val_filepath_all.extend(val_filepaths)
        test_filepath_all.extend(test_filepaths)

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("./tokenizer_change")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model = model.to("cuda")

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
        if len(set(speaker_label)) > MAX_SPEAKER:
            continue
        df1, tder, acc_u = evaluate_conversation(model, tokenizer, conversation, speaker_label, min_sentence_num, max_sentence_num)
        sen_num = len(conversation)
        word_num = sum([len(sentence.split()) for sentence in conversation])
        print(f"filepath: {filepath}\nDF1: {df1}, TDER: {tder}, ACC_U: {acc_u}, sentence num: {sen_num} word num: {word_num}")
        tder_list.append(tder)
        df1_list.append(df1)
        acc_u_list.append(acc_u)
        sen_num_list.append(sen_num)
        word_num_list.append(word_num)
    print("\n==================================================================================")
    print(f"Checkpoint: {checkpoint}")
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
    for filepath in test_filepath_all:
        if os.path.splitext(filepath)[1] == ".json":
            with open(filepath, 'r') as json_in:
                content = json.load(json_in)  # fill in the content of conversation in this format
        conversation, speaker_label = preprocess_conversation(content)
        if len(set(speaker_label)) > MAX_SPEAKER:
            continue
        df1, tder, acc_u = evaluate_conversation(model, tokenizer, conversation, speaker_label, min_sentence_num, max_sentence_num)
        sen_num = len(conversation)
        word_num = sum([len(sentence.split()) for sentence in conversation])
        print(f"filepath: {filepath}\nDF1: {df1}, TDER: {tder}, ACC_U: {acc_u}, sentence num: {sen_num} word num: {word_num}")
        tder_list.append(tder)
        df1_list.append(df1)
        acc_u_list.append(acc_u)
        sen_num_list.append(sen_num)
        word_num_list.append(word_num)
    print("\n==================================================================================")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test avg DF1: {sum(df1_list) / len(df1_list)}, Test avg TDER: {sum(tder_list) / len(tder_list)}, Test avg acc utterance: {sum(acc_u_list) / len(acc_u_list)}")
    total_weights = sum(sen_num_list)
    weighted_tder_sum = sum(t * w for t, w in zip(tder_list, sen_num_list))
    weighted_df1_sum = sum(d * w for d, w in zip(df1_list, sen_num_list))
    weighted_acc_u_sum = sum(a * w for a, w in zip(acc_u_list, sen_num_list))
    print(f"(Sentence-level Weighted) Test avg DF1: {weighted_df1_sum / total_weights}, Test avg TDER: {weighted_tder_sum / total_weights}, Test avg acc utterance: {weighted_acc_u_sum / total_weights}")
    total_weights = sum(word_num_list)
    weighted_tder_sum = sum(t * w for t, w in zip(tder_list, word_num_list))
    weighted_df1_sum = sum(d * w for d, w in zip(df1_list, word_num_list))
    weighted_acc_u_sum = sum(a * w for a, w in zip(acc_u_list, word_num_list))
    print(f"(Word-level Weighted) Test avg DF1: {weighted_df1_sum / total_weights}, Test avg TDER: {weighted_tder_sum / total_weights}, Test avg acc utterance: {weighted_acc_u_sum / total_weights}")
    print("==================================================================================\n")


if __name__ == "__main__":
    # evaluate_checkpoint("./results/t5-3b-d7-0-sd-8-6e5/checkpoint-257398", 2, 8)
    evaluate_checkpoint("./results/t5-3b-d7-0-sd-8-6e5/checkpoint-386097", 2, 8)
    evaluate_checkpoint("./results/t5-3b-d7-0-sd-8-1e4/checkpoint-257398", 2, 8)
    evaluate_checkpoint("./results/t5-3b-d7-0-sd-8-1e4/checkpoint-386097", 2, 8)
