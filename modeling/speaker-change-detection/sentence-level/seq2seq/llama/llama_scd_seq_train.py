import json
import os
import string
import random

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Hyper parameters
SEED = 42
INPUT_MAX_LENGTH = 512
OUTPUT_MAX_LENGTH = 512
MAX_SPEAKER = float("inf")
BATCH_SIZE = 8
EPOCHS = 3
CHANGE_POINT = " <change> "

hf_token = "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"  # REMOVE THIS BEFORE COMMIT & PUSH!!!


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []
    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict)
        output_list.append(unique_dict[item])
    return output_list


def format_input_output(data_dir: str, min_sentence_num: int = 2, max_sentence_num: int | float = float("inf")):
    with open(data_dir, 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]
    conversation_list = []
    speaker_change_list = []
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
                if len(tokenizer.encode(input_text)) > INPUT_MAX_LENGTH:
                    print(f"input length over max, max: {INPUT_MAX_LENGTH}, actual: {len(tokenizer.encode(input_text))}")
                if len(tokenizer.encode(output_text)) > OUTPUT_MAX_LENGTH:
                    print(f"input length over max, max: {OUTPUT_MAX_LENGTH}, actual: {len(tokenizer.encode(output_text))}")
                conversation_list.append(input_text)
                speaker_change_list.append(output_text)
    return conversation_list, speaker_change_list


def map_function(batch):
    input_ids = tokenizer.apply_chat_template(
        batch["chat"],
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        max_length=INPUT_MAX_LENGTH,
        truncation=True
    )
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


if __name__ == "__main__":
    seed_everything(SEED)
    # Load tokenizer and model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./tokenizers", token=hf_token)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.save_pretrained("./tokenizer_pad")
    # tokenizer = AutoTokenizer.from_pretrained("./tokenizer_change")
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="./models", token=hf_token)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create dataset and dataloader
    # data_train_dir = "/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json"
    data_train_dir = "/local/scratch/pwu54/Text-based SD Dataset/dataset7_align_train_sent_2sp.json"
    conversation_list, speaker_change_list = format_input_output(data_train_dir, 2, 8)

    chat_list = []
    for conversation, speaker_change in zip(conversation_list, speaker_change_list):
        system_message = "You are a helpful, respectful, and honest assistant."
        user_message = f"""Determine whether the speaker is changed or not for each point marked by{CHANGE_POINT}in the 
        conversation. Give a sequence of 0 and 1 as output, 0 for unchanged, 1 for changed.\n\nConversation: {conversation}"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": speaker_change}
        ]
        chat_list.append(messages)
    dataset_train = Dataset.from_dict({"chat": chat_list})
    dataset_train = dataset_train.map(
        map_function,
        # batched=True,
        num_proc=8
    )
    dataset_train = dataset_train.shuffle(seed=SEED)

    # Define Training Arguments and Initialize Trainer
    training_args = TrainingArguments(
        output_dir='./results/llama3-8b-d7-scd-28-5e5',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        optim="adamw_torch",
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        bf16=True,
        save_strategy="epoch",
        deepspeed="./deepspeed_config_zero2_offloadopt.json",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors='pt', pad_to_multiple_of=8, padding=True)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        data_collator=collator
    )

    # Train the Model
    model.train()
    trainer.train()
