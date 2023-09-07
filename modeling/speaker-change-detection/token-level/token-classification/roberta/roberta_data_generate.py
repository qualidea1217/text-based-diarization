import json
import os
from multiprocessing import Pool

from transformers import BertTokenizer

MAX_LENGTH = 512
STRIDE = 64

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', cache_dir="./tokenizers")


def process_chunk(args):
    conversations_chunk, speaker_labels_chunk = args
    texts = []
    labels = []
    finished = 0
    finished_percentage = 0
    for conversation, speaker_label in zip(conversations, speaker_labels):
        tokenized_conversation = []
        label = []
        for utterance in conversation:
            tokenized_utterance = tokenizer.tokenize(utterance)
            tokenized_conversation.extend(tokenized_utterance)
            label.extend([0] * len(tokenized_utterance))
            last_token_left = 2
            for i in range(len(label) - 1, -1, -1):
                if tokenized_conversation[i][0] != "##":
                    label[i] = 1
                    last_token_left -= 1
                if last_token_left <= 0:
                    break
        chunk_size = MAX_LENGTH - 2  # exclude the [CLS] and [SEP] special token
        for i in range(0, len(tokenized_conversation) - chunk_size + 1, STRIDE):
            texts.append(tokenized_conversation[i:i + chunk_size])
            labels.append(label[i:i + chunk_size])
        finished += 1
        if finished >= finished_percentage * (len(conversations_chunk) / 100):
            print(f"process: {os.getpid()} {finished_percentage}% completed.")
            finished_percentage += 1
    return texts, labels


if __name__ == "__main__":
    with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_utterance.json", 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = data_dict["speaker_list"]

    num_cores = 16
    chunk_size = len(conversations) // num_cores
    data_chunks = [(conversations[i:i + chunk_size], speaker_labels[i:i + chunk_size]) for i in
                   range(0, len(conversations), chunk_size)]

    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, data_chunks)

    texts = []
    labels = []
    for res in results:
        texts.extend(res[0])
        labels.extend(res[1])

    with open("interview_roberta_scd_tkc_512.json", 'w') as json_out:
        json.dump({"text": texts, "label": labels}, json_out, indent=4)

