import json

import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []
    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict)
        output_list.append(unique_dict[item])
    return output_list


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["conversations"], padding="max_length", truncation=True, max_length=ENCODER_MAX_LENGTH)
    outputs = tokenizer(batch["speaker_labels"], padding="max_length", truncation=True, max_length=DECODER_MAX_LENGTH)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch


def preprocess_data(data_dir: str, min_sentence_num: int = 1, max_sentence_num: int | float = float("inf")):
    with open(data_dir, 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]
    input_list = []
    output_list = []
    for conversation, speaker_label in tqdm(zip(conversations, speaker_labels), total=len(conversations)):
        for i in range(len(conversation)):
            for j in range(i + min_sentence_num, len(conversation) + 1):
                if j - i > max_sentence_num:
                    break
                input_text = CHANGE_POINT.join([sentence for sentence in conversation[i:j]])
                output_text = " ".join(["1" if speaker_label[i:j][k] != speaker_label[i:j][k - 1] else "0" for k in range(1, len(speaker_label[i:j]))])
                if len(tokenizer.encode(input_text)) > ENCODER_MAX_LENGTH:
                    break
                input_list.append(input_text)
                output_list.append(output_text)
    return input_list, output_list


def predict_single_input(model, tokenizer, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    return list(map(int, decoded_output.split()))


# Hyper parameters
ENCODER_MAX_LENGTH = 512
DECODER_MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 3
CHANGE_POINT = " <change> "

if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("./tokenizer_bos")
    model = T5ForConditionalGeneration.from_pretrained("./results/t5-3b/checkpoint-9494")
    model = model.to("cuda")

    conversation_test = ["Do you have any plans for the weekend?",
                         "Not really, I was thinking of maybe catching a movie or going for a hike.",
                         "How about you?", "A hike sounds great!",
                         "There's a scenic trail nearby I've been wanting to explore.", "Would you like to join me?"]
    for i in range(len(conversation_test)):
        for j in range(i + 2, len(conversation_test) + 1):
            if j - i > 4:
                break
            input_text = CHANGE_POINT.join([sentence for sentence in conversation_test[i:j]])
            result = predict_single_input(model, tokenizer, input_text)
            print(result)
