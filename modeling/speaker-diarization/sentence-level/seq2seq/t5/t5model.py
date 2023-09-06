import json

from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []
    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict) + 1  # Add 1 here
        output_list.append(unique_dict[item])
    return output_list


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["conversations"],
        padding="max_length",
        truncation=True,
        max_length=ENCODER_MAX_LENGTH,
    )
    outputs = tokenizer(
        batch["speaker_labels"],
        padding="max_length",
        truncation=True,
        max_length=DECODER_MAX_LENGTH,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


# Hyper parameters
ENCODER_MAX_LENGTH = 1024
DECODER_MAX_LENGTH = 1024
BATCH_SIZE = 1
EPOCHS = 5

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-3b', cache_dir="./tokenizers", model_max_length=ENCODER_MAX_LENGTH)
model = T5ForConditionalGeneration.from_pretrained('t5-3b', cache_dir="./models")

with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
    data_dict = json.load(json_in)
    conversations = data_dict["text_list"]
    speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]

conversations = ["classify: " + " ".join(conv) for conv in conversations]
speaker_labels = [" ".join(map(str, label_seq)) for label_seq in speaker_labels]

conversations_filtered = []
speaker_labels_filtered = []
for i in range(len(conversations)):
    if len(tokenizer.encode(conversations[i])) <= ENCODER_MAX_LENGTH and len(
            tokenizer.encode(speaker_labels[i])) <= DECODER_MAX_LENGTH:
        conversations_filtered.append(conversations[i])
        speaker_labels_filtered.append(speaker_labels[i])

custom_dataset = Dataset.from_dict({"conversations": conversations_filtered, "speaker_labels": speaker_labels_filtered})
custom_dataset = custom_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

# Create train/test dataset and dataloader
dataset_train = custom_dataset["train"]
dataset_train = dataset_train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["conversations", "speaker_labels"],
)

dataset_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

dataset_test = custom_dataset["test"]
dataset_test = dataset_test.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["conversations", "speaker_labels"],
)

dataset_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# 3. Define Training Arguments and Initialize Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir='./results/t5-3b-1024',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    optim="adafactor",
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    bf16=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    eval_accumulation_steps=1
)


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pred_list = [list(map(int, s.split())) for s in pred_str]
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    label_list = [list(map(int, l.split())) for l in label_str]

    acc_list = []
    for i in range(len(pred_list)):
        correct = 0
        for j in range(min(len(pred_list[i]), len(label_list[i]))):
            if pred_list[i][j] == label_list[i][j]:
                correct += 1
        acc_list.append(correct / len(label_list[i]))
    return {"acc_list_mean": sum(acc_list) / len(acc_list)}


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics
)

# 4. Train the Model
trainer.train()


# 5. Inference
def predict_speaker_sequence(model, tokenizer, conversation):
    input_text = "classify: " + " ".join(conversation)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    return list(map(int, decoded_output.split()))


conversation_test = ["Hey, are you available?", "Yes, what's up?", "Let's discuss the project."]
predicted_sequence = predict_speaker_sequence(model, tokenizer, conversation_test)
print(predicted_sequence)
