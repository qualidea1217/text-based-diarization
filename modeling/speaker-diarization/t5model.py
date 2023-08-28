import json

from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []

    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict) + 1  # Add 1 here
        output_list.append(unique_dict[item])

    return output_list


# 1. Data Representation & Preprocessing
with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
    data_dict = json.load(json_in)
    conversations = data_dict["text_list"]
    labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]

texts = [" ".join(conv) for conv in conversations]
label_strs = [" ".join(map(str, label_seq)) for label_seq in labels]


# 2. Custom Dataset
class T5DiarizationDataset(Dataset):
    def __init__(self, texts, label_strs, tokenizer, max_length=1024):
        # Filter the data right here based on max_length
        filtered_texts = []
        filtered_labels = []

        for text, label in zip(texts, label_strs):
            encoded_text = tokenizer.encode(f"classify: {text}", add_special_tokens=True)
            encoded_label = tokenizer.encode(label, add_special_tokens=True)

            if len(encoded_text) <= max_length and len(encoded_label) <= max_length:
                # You may want to set different lengths for text and label if necessary
                filtered_texts.append(text)
                filtered_labels.append(label)

        self.texts = filtered_texts
        self.label_strs = filtered_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_str = self.label_strs[idx]

        inputs = self.tokenizer.encode_plus(
            f"classify: {text}",
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        label_ids = self.tokenizer.encode(
            label_str,
            add_special_tokens=True,
            max_length=self.max_length,
            # Use a suitable max_length for labels. It may not be the same as the input max_length.
            pad_to_max_length=True,
            truncation=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small', cache_dir="./tokenizers")
model = T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir="./models")

# Create dataset and dataloader
dataset = T5DiarizationDataset(texts, label_strs, tokenizer)
print(len(dataset))

# 3. Define Training Arguments and Initialize Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 4. Train the Model
trainer.train()


# 5. Inference
def predict_speaker_sequence(model, tokenizer, conversation):
    input_text = "classify: " + " ".join(conversation)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return list(map(int, decoded_output.split()))


conversation_test = ["Hey, are you available?", "Yes, what's up?", "Let's discuss the project."]
predicted_sequence = predict_speaker_sequence(model, tokenizer, conversation_test)
print(predicted_sequence)
