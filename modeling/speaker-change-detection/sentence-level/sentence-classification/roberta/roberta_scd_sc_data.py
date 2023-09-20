import json
import os
from multiprocessing import Pool

from transformers import RobertaTokenizer


MAX_LENGTH = 512
SEPARATION = " </s> </s> "
HISTORY_SEPARATION = " "
HISTORY_UTTERANCE_START = " "
FUTURE_SEPARATION = " "
FUTURE_START = " "

tokenizer = RobertaTokenizer.from_pretrained("roberta-large", cache_dir="./tokenizers")
# If new special tokens are added, remember to add it to the tokenizer and resize the model during data process and before training


def get_history_sequence(history: list, history_speaker: list) -> str:
    if len(history) == 1:
        return history[0]
    history_sequence = HISTORY_UTTERANCE_START + history[0]
    for i in range(1, len(history)):
        if history_speaker[i] == history_speaker[i - 1]:
            history_sequence += " " + history[i]
        else:
            history_sequence += HISTORY_UTTERANCE_START + history[i]
    return history_sequence


def process_chunk(args):
    conversations_chunk, speaker_labels_chunk, history_utterance_num, future_sentence_num = args
    texts = []
    labels = []
    finished = 0
    finished_percentage = 0
    for conversation, speaker_label in zip(conversations_chunk, speaker_labels_chunk):
        for i in range(1, len(conversation) - future_sentence_num):
            target = conversation[i]
            history = [conversation[i - 1]]
            history_speaker = [speaker_label[i - 1]]
            if future_sentence_num == 0:
                future = ""
            else:
                future = FUTURE_START + " ".join(conversation[i + 1:i + 1 + future_sentence_num])

            input_context = conversation[i - 1] + SEPARATION + target + future
            speaker_change = 0 if speaker_label[i - 1] == speaker_label[i] else 1

            if len(tokenizer.encode(input_context)) > MAX_LENGTH:
                continue

            for j in range(i - 2, -1, -1):
                if history_utterance_num == 0:  # if equals to 0, not consider utterance and fill until max length
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
                    if speaker_change > history_utterance_num:  # Specify the number of utterance in history
                        break
                    # If utterance is not fill out, continue to add sentence to history
                    history.insert(0, conversation[j])
                    history_speaker.insert(0, speaker_label[j])
            # Set label according to speaker change or not (0 for unchange, 1 for change)
            input_context = get_history_sequence(history, history_speaker) + SEPARATION + target + future
            if len(tokenizer.encode(input_context)) <= MAX_LENGTH:
                label = 0 if speaker_label[i - 1] == speaker_label[i] else 1
                texts.append(input_context.strip())  # remove white space at begin and end
                labels.append(label)
        finished += 1
        if finished >= finished_percentage * (len(conversations_chunk) / 100):
            print(f"process: {os.getpid()} {finished_percentage}% completed.")
            finished_percentage += 1
    return texts, labels


def generate_json_data(input_dir: str, output_dir: str, history_utterance_num: int, future_sentence_num: int):
    with open(input_dir, 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = data_dict["speaker_list"]

    # Split data into chunks for parallel processing
    num_cores = 8
    chunk_size = len(conversations) // num_cores

    args = [
        (conversations[i:i + chunk_size], speaker_labels[i:i + chunk_size], history_utterance_num, future_sentence_num)
        for i in range(0, len(conversations), chunk_size)
    ]

    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, args)

    texts = []
    labels = []
    for res in results:
        texts.extend(res[0])
        labels.extend(res[1])

    with open(output_dir, 'w') as json_out:
        json.dump({"text": texts, "label": labels}, json_out, indent=4)


if __name__ == "__main__":
    # Generate train data
    generate_json_data("/local/scratch/pwu54/Text-based SD Dataset/dataset7_align_train_sent.json",
                       "/local/scratch/pwu54/Text-based SD Dataset/dataset7_roberta_scd_512_train.json",
                       0, 0)

    # Generate validation data
    generate_json_data("/local/scratch/pwu54/Text-based SD Dataset/dataset7_align_val_sent.json",
                       "/local/scratch/pwu54/Text-based SD Dataset/dataset7_roberta_scd_512_val.json",
                       0, 0)

    # Generate test data
    generate_json_data("/local/scratch/pwu54/Text-based SD Dataset/dataset7_align_test_sent.json",
                       "/local/scratch/pwu54/Text-based SD Dataset/dataset7_roberta_scd_512_test.json",
                       0, 0)
