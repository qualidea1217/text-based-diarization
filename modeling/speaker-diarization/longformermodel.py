import json

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []
    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict) + 1  # Add 1 here
        output_list.append(unique_dict[item])
    return output_list


def process_data_to_model_inputs(batch, encoder_max_length, decoder_max_length):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["conversations"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["speaker_labels"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# Hyper parameters
ENCODER_MAX_LENGTH = 8192
DECODER_MAX_LENGTH = 8192
BATCH_SIZE = 4
EPOCHS = 5

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/led-large-16384', cache_dir="./tokenizers", model_max_length=ENCODER_MAX_LENGTH)
model = AutoModelForSeq2SeqLM.from_pretrained('allenai/led-large-16384', cache_dir="./models", gradient_checkpointing=True)

# Data Preprocessing
with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
    data_dict = json.load(json_in)
    conversations = data_dict["text_list"]
    speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]

conversations = [" ".join(conv) for conv in conversations]
speaker_labels = [" ".join(map(str, label_seq)) for label_seq in speaker_labels]
custom_dataset = Dataset.from_dict({"conversations": conversations, "speaker_labels": speaker_labels})

# Create dataset and dataloader
dataset_train = custom_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["conversations", "speaker_labels"],
)

dataset_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
print(len(dataset_train))

# 3. Define Training Arguments and Initialize Trainer
training_args = TrainingArguments(
    output_dir='./results/longformer',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    optim="adafactor",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train
)

# 4. Train the Model
trainer.train()


# 5. Inference
def predict_speaker_sequence(model, tokenizer, conversation):
    input_text = " ".join(conversation)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    return list(map(int, decoded_output.split()))


conversation_test = ["Hey, are you available?", "Yes, what's up?", "Let's discuss the project."]
predicted_sequence = predict_speaker_sequence(model, tokenizer, conversation_test)
print(predicted_sequence)



