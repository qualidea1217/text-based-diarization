from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW
from torch.utils.data import Dataset, DataLoader
import torch

from preprocessing.pure_text_transcript import get_interview_dataset

# 1. Data Representation & Preprocessing
conversations = [
    ["Hello, how are you?", "I'm good, thanks!", "That's great!"],
    ["Is this seat taken?", "No, you can sit here.", "Thanks!"]
]

labels = [
    [1, 2, 1],
    [1, 2, 1]
]

texts = [" ".join(conv) for conv in conversations]
label_strs = [" ".join(map(str, label_seq)) for label_seq in labels]


# 2. Custom Dataset and DataLoader
class T5DiarizationDataset(Dataset):
    def __init__(self, texts, label_strs, tokenizer, max_length):
        self.texts = texts
        self.label_strs = label_strs
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

        label_ids = self.tokenizer.encode(label_str, add_special_tokens=False)

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# 3. Model Loading, Fine-tuning, and Training
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

dataset = T5DiarizationDataset(texts, label_strs, tokenizer, max_length=4096)
loader = DataLoader(dataset, batch_size=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

EPOCHS = 3

model.train()
for epoch in range(EPOCHS):
    for batch in loader:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()


# 4. Evaluation and Inference
def predict_speaker_sequence(model, tokenizer, conversation):
    input_text = "classify: " + " ".join(conversation)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return list(map(int, decoded_output.split()))


conversation_test = ["Hey, are you available?", "Yes, what's up?", "Let's discuss the project."]
predicted_sequence = predict_speaker_sequence(model, tokenizer, conversation_test)
print(predicted_sequence)  # Might output something like [1, 2, 1] depending on the training
