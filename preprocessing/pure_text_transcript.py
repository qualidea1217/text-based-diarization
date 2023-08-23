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


def get_interview_dataset_for_token_classification(tokenizer) -> tuple[datasets.Dataset, int]:
    df = pd.read_csv("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/utterances.csv")

    # Lists to store the tokenized data
    texts = []
    labels_list = []
    attention_masks = []
    max_speaker = 2

    for episode in df['episode'].unique():
        df_episode = df[df["episode"] == episode].sort_values(by="episode_order")

        # Lists for chunks
        chunk_texts = []
        chunk_labels = []

        for _, row in df_episode.iterrows():
            # Local dictionary for speaker IDs within the chunk
            speaker_dict = {}
            current_id = 0

            # Tokenize the current utterance (only input_ids, not full encoding)
            try:
                input_ids = tokenizer.encode(row['utterance'], add_special_tokens=False)  # Avoid special tokens since we're concatenating
                if len(input_ids) > 512:
                    continue
            except:
                print(row['utterance'])
                continue

            # If adding the new tokens will exceed the max length, save the current chunk and start a new one
            if len(chunk_texts) + len(input_ids) > 512:
                texts.append(tokenizer.decode(chunk_texts))
                labels_list.append(chunk_labels)
                attention_masks.append([1] * len(chunk_texts))

                # Reset chunk lists and speaker dictionary for the new chunk
                chunk_texts = []
                chunk_labels = []
                speaker_dict = {}
                current_id = 0

            # Get the current speaker and update the speaker dictionary
            speaker = row['speaker']
            if speaker not in speaker_dict:
                speaker_dict[speaker] = current_id
                current_id += 1
                if max_speaker < current_id + 1:
                    max_speaker += 1

            # Extend chunk_texts and chunk_labels
            chunk_texts.extend(input_ids)
            chunk_labels.extend([speaker_dict[speaker]] * len(input_ids))

        # Save any remaining tokens as a new chunk
        if chunk_texts:
            texts.append(tokenizer.decode(chunk_texts))
            labels_list.append(chunk_labels)
            attention_masks.append([1] * len(chunk_texts))

    # Construct dataset from lists
    dataset = Dataset.from_dict({
        'text': texts,
        'labels': labels_list,
        'attention_mask': attention_masks
    })

    return dataset, max_speaker


def get_interview_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", cache_dir="./tokenizers")
    df = pd.read_csv("D:/Text-based SD Dataset/INTERVIEW/utterances.csv")
    df = df.sort_values(by=["episode", "episode_order"])
    result = df.groupby(by="episode").agg(lambda x: list(x)).reset_index()
    df.to_csv("INTERVIEW.csv")
    # for episode in df['episode'].unique():
    #     if episode == 1:
    #         utterances = []
    #         speaker_ids = []
    #         df_episode = df[df["episode"] == episode].sort_values(by="episode_order")
    #         for _, row in df_episode.iterrows():
    #             utterances.append(row["utterance"])
    #             speaker_ids.append(row["speaker"])
    #         print(len(set(speaker_ids)))
    #         print(len(' '.join(utterances).split()))


if __name__ == "__main__":
    get_interview_dataset()