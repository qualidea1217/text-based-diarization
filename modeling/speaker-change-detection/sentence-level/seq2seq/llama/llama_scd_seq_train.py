import json
import string

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hyper parameters
MAX_LENGTH = 512  # decoder-only model requires input and output having same length
MAX_SPEAKER = float("inf")
BATCH_SIZE = 8
EPOCHS = 3
CHANGE_POINT = " <change> "

hf_token = ""  # REMOVE THIS BEFORE COMMIT & PUSH!!!


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
    inputs = tokenizer(batch["conversations"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    outputs = tokenizer(batch["speaker_labels"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch


def preprocess_data(data_dir: str, min_sentence_num: int = 2, max_sentence_num: int | float = float("inf")):
    with open(data_dir, 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]
    input_list = []
    output_list = []
    for conversation, speaker_label in tqdm(zip(conversations, speaker_labels), total=len(conversations)):
        conversation_filter = [conversation[i] for i in range(len(conversation)) if
                            conversation[i] not in string.punctuation and conversation[i] not in string.whitespace]
        speaker_label_filter = [speaker_label[i] for i in range(len(speaker_label)) if
                               conversation[i] not in string.punctuation and conversation[i] not in string.whitespace]
        conversation, speaker_label = conversation_filter, speaker_label_filter
        for i in range(len(conversation)):
            for j in range(i + min_sentence_num, len(conversation) + 1):
                if len(set(speaker_label[i:j])) > MAX_SPEAKER:
                    break
                if j - i > max_sentence_num:
                    break
                input_text = CHANGE_POINT.join([sentence for sentence in conversation[i:j]])
                output_text = ''.join(["1" if speaker_label[i:j][k] != speaker_label[i:j][k - 1] else "0"
                                        for k in range(1, len(speaker_label[i:j]))])
                if len(tokenizer.encode(input_text)) > MAX_LENGTH:
                    print(f"input length over max, max: {MAX_LENGTH}, actual: {len(tokenizer.encode(input_text))}")
                    continue
                if len(tokenizer.encode(output_text)) > MAX_LENGTH:
                    print(f"input length over max, max: {MAX_LENGTH}, actual: {len(tokenizer.encode(output_text))}")
                    continue
                input_list.append(input_text)
                output_list.append(output_text)
    return input_list, output_list


def preprocess_data_single_pred(data_dir: str, min_sentence_num: int = 2, max_sentence_num: int | float = float("inf")):
    with open(data_dir, 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]
    input_list = []
    output_list = []
    for conversation, speaker_label in tqdm(zip(conversations, speaker_labels), total=len(conversations)):
        conversation_filter = [conversation[i] for i in range(len(conversation)) if
                            conversation[i] not in string.punctuation and conversation[i] not in string.whitespace]
        speaker_label_filter = [speaker_label[i] for i in range(len(speaker_label)) if
                               conversation[i] not in string.punctuation and conversation[i] not in string.whitespace]
        conversation, speaker_label = conversation_filter, speaker_label_filter
        for i in range(len(conversation)):
            for j in range(i + min_sentence_num, len(conversation) + 1):
                if len(set(speaker_label[i:j])) > 2:
                    break
                if j - i > max_sentence_num:
                    break
                input_text = CHANGE_POINT.join([sentence for sentence in conversation[i:j]])
                input_text = " ".join(input_text.split(CHANGE_POINT)[:-1]) + CHANGE_POINT + \
                             input_text.split(CHANGE_POINT)[-1]
                output_text = ["1" if speaker_label[i:j][k] != speaker_label[i:j][k - 1] else "0" for k in
                               range(1, len(speaker_label[i:j]))][-1]
                if len(tokenizer.encode(input_text)) > MAX_LENGTH:
                    break
                input_list.append(input_text)
                output_list.append(output_text)
    return input_list, output_list


if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="./tokenizers", model_max_length=MAX_LENGTH, token=hf_token)
    tokenizer.add_special_tokens({"pad_token": "<pad>", "additional_special_tokens": [CHANGE_POINT]})
    tokenizer.save_pretrained("./tokenizer_change")
    # tokenizer = AutoTokenizer.from_pretrained("./tokenizer_change")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="./models", token=hf_token)
    model.resize_token_embeddings(len(tokenizer))

    # Create dataset and dataloader
    # data_train_dir = "/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json"
    data_train_dir = "/local/scratch/pwu54/Text-based SD Dataset/dataset7_align_train_sent_2sp.json"
    input_train, output_train = preprocess_data(data_train_dir, 2, 8)
    dataset_train = Dataset.from_dict({"conversations": input_train, "speaker_labels": output_train})
    dataset_train = dataset_train.map(
        process_data_to_model_inputs,
        batched=True,
        num_proc=8,
        remove_columns=["conversations", "speaker_labels"],
    )

    dataset_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 3. Define Training Arguments and Initialize Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results/llama2-7b-d7-scd-28-1e4',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        optim="adamw_torch",
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        bf16=True,
        save_strategy="epoch",
        deepspeed="./deepspeed_config_zero2_offloadopt.json",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
    )

    # 4. Train the Model
    model.train()
    trainer.train()
