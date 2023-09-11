import json

from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments


# Hyper parameters
ENCODER_MAX_LENGTH = 1024
DECODER_MAX_LENGTH = 1024
UNCHANGE_SPECIAL_TOKEN = " "
CHANGE_SPECIAL_TOKEN = "<change>"
BATCH_SIZE = 2
EPOCHS = 5

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-3b', cache_dir="./tokenizers", model_max_length=ENCODER_MAX_LENGTH)
model = T5ForConditionalGeneration.from_pretrained("./results/t5-3b-1024/checkpoint-14726")
model.to("cuda")

# Add special tokens (optional)
# tokenizer.add_special_tokens({"additional_special_tokens": CHANGE_SPECIAL_TOKEN})
# model.resize_token_embeddings(len(tokenizer))

# Load data
with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_utterance.json", 'r') as json_in:
    data_dict = json.load(json_in)
    conversations = data_dict["text_list"]
    speaker_labels = data_dict["speaker_list"]

# Add special token for marking change
texts = []
labels = []
for i in range(len(conversations)):
    conversation = conversations[i][0]
    for j in range(1, len(conversations[i])):
        if speaker_labels[i][j] == speaker_labels[i][j - 1]:
            conversation += " " + UNCHANGE_SPECIAL_TOKEN + " " + conversations[i][j]
        else:
            conversation += CHANGE_SPECIAL_TOKEN + conversations[i][j]
    labels.append(conversation)
    texts.append(" ".join(conversations[i]))

# Remove all data point with length larger than max length
texts_filtered = []
labels_filtered = []
for i in range(len(texts)):
    if len(tokenizer.encode(texts[i])) <= ENCODER_MAX_LENGTH and len(tokenizer.encode(labels[i])) <= DECODER_MAX_LENGTH:
        texts_filtered.append(texts[i])
        labels_filtered.append(labels[i])

custom_dataset = Dataset.from_dict({"text": texts_filtered, "label": labels_filtered})
custom_dataset = custom_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
dataset_test = custom_dataset["test"]


def preprocess_function(batch):
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=ENCODER_MAX_LENGTH)
    outputs = tokenizer(batch["label"], padding="max_length", truncation=True, max_length=DECODER_MAX_LENGTH)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]for labels in batch["labels"]]
    return batch


# Preprocess the dataset
dataset_test = dataset_test.map(preprocess_function, batched=True, batch_size=BATCH_SIZE, remove_columns=["text", "label"])
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], device="cuda")

# Give output from test
for i in range(len(dataset_test["input_ids"])):
    output = model.generate(input_ids=dataset_test['input_ids'][i].unsqueeze(0), attention_mask=dataset_test['attention_mask'][i].unsqueeze(0), max_length=DECODER_MAX_LENGTH)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    print(prediction)
# outputs = model.generate(input_ids=dataset_test['input_ids'], attention_mask=dataset_test['attention_mask'], max_length=DECODER_MAX_LENGTH)
# predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
# for i in range(len(predictions)):
#     print(predictions[i])