import json
import os
from multiprocessing import Pool

from transformers import BertTokenizer

MAX_LENGTH = 512
STRIDE = 64

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', cache_dir="./tokenizers")


if __name__ == "__main__":
    with open("interview_utterance.json", 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = data_dict["speaker_list"]

    texts = []
    labels = []

    for conversation, speaker_label in zip(conversations, speaker_labels):
        tokenized_conversation = []
        label = []
        for utterance in conversation:
            tokenized_utterance = tokenizer.tokenize(utterance)
            tokenized_conversation.extend(tokenized_utterance)
            label.extend([0] * len(tokenized_utterance))
            last_token_left = 2
            for i in range(len(label) - 1, -1, -1):
                if tokenized_conversation[i][:2] != "##":
                    label[i] = 1
                    last_token_left -= 1
                if last_token_left <= 0:
                    break
        chunk_size = MAX_LENGTH - 2  # exclude the [CLS] and [SEP] special token
        print(len(tokenized_conversation))
        for i in range(0, len(tokenized_conversation) - chunk_size + 1, STRIDE):
            texts.append(tokenized_conversation[i:i + chunk_size])
            labels.append(label[i:i + chunk_size])
            print(len(texts[-1]))
            print(i + chunk_size)
        break

