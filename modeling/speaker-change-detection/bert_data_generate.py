import json
import os
from multiprocessing import Pool

from transformers import BertTokenizer

MAX_LENGTH = 512
HISTORY_SEPARATION = " [SEP] "
HISTORY_LAST_SEPARATION = " [SEP] "
TARGET_SEPARATION = " [SEP] "

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', cache_dir="./tokenizers")


def process_data():
    with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = data_dict["speaker_list"]

    texts = []
    labels = []

    # Convert raw data to formatted data with length limitation
    for conversation, speaker_label in zip(conversations, speaker_labels):
        for i in range(1, len(conversation)):
            input_context = conversation[i - 1] + TARGET_SEPARATION + conversation[i]
            if len(tokenizer.encode(input_context)) > MAX_LENGTH:
                continue
            for j in range(i - 2, -1, -1):
                if j == i - 2:
                    input_context_temp = conversation[j] + HISTORY_LAST_SEPARATION + input_context
                else:
                    input_context_temp = conversation[j] + HISTORY_SEPARATION + input_context
                if len(tokenizer.encode(input_context_temp)) > MAX_LENGTH:
                    break
                else:
                    input_context = input_context_temp
            label = 1 if speaker_label[i - 1] == speaker_label[i] else 0
            texts.append(input_context)
            labels.append(label)
    return texts, labels


# Function to preprocess a chunk of data
def process_chunk(args):
    conversations_chunk, speaker_labels_chunk = args
    texts = []
    labels = []
    finished = 0
    finished_percentage = 0
    for conversation, speaker_label in zip(conversations_chunk, speaker_labels_chunk):
        for i in range(1, len(conversation)):
            input_context = conversation[i - 1] + TARGET_SEPARATION + conversation[i]
            if len(tokenizer.encode(input_context)) > MAX_LENGTH:
                continue
            for j in range(i - 2, -1, -1):
                if j == i - 2:
                    input_context_temp = conversation[j] + HISTORY_LAST_SEPARATION + input_context
                else:
                    input_context_temp = conversation[j] + HISTORY_SEPARATION + input_context
                if len(tokenizer.encode(input_context_temp)) > MAX_LENGTH:
                    break
                else:
                    input_context = input_context_temp
            label = 1 if speaker_label[i - 1] == speaker_label[i] else 0
            texts.append(input_context)
            labels.append(label)
        finished += 1
        if finished >= finished_percentage * (len(conversations_chunk) / 100):
            print(f"process: {os.getpid()} {finished_percentage}% completed.")
            finished_percentage += 1
    return texts, labels


if __name__ == "__main__":
    # Load raw data
    with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = data_dict["speaker_list"]

    # Split data into chunks for parallel processing
    num_cores = 16
    chunk_size = len(conversations) // num_cores

    data_chunks = [(conversations[i:i + chunk_size], speaker_labels[i:i + chunk_size]) for i in
                   range(0, len(conversations), chunk_size)]

    # Process data in parallel
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, data_chunks)

    texts = []
    labels = []
    for res in results:
        texts.extend(res[0])
        labels.extend(res[1])

    with open("bert_dataset_512.json", 'w') as json_out:
        json.dump({"text": texts, "label": labels}, json_out, indent=4)
