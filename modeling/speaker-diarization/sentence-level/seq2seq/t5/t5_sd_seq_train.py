import json
from multiprocessing import Pool
from tqdm import tqdm

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
                input_text = "".join([BEGIN_OF_SENTENCE + sentence for sentence in conversation[i:j]])
                output_text = " ".join([str(speaker) for speaker in speaker_label[i:j]])
                if len(tokenizer.encode(input_text)) > ENCODER_MAX_LENGTH:
                    break
                input_list.append(input_text)
                output_list.append(output_text)
    return input_list, output_list


def preprocess_data_chunk(args):
    conversations_chunk, speaker_labels_chunk, min_sentence_num, max_sentence_num = args
    input_list = []
    output_list = []
    for conversation, speaker_label in tqdm(zip(conversations_chunk, speaker_labels_chunk), total=len(conversations_chunk)):
        for i in range(len(conversation)):
            for j in range(i + min_sentence_num, len(conversation) + 1):
                if j - i > max_sentence_num:
                    break
                input_text = "".join([BEGIN_OF_SENTENCE + sentence for sentence in conversation[i:j]])
                output_text = " ".join([str(speaker) for speaker in speaker_label[i:j]])
                if len(tokenizer.encode(input_text)) > ENCODER_MAX_LENGTH:
                    break
                input_list.append(input_text)
                output_list.append(output_text)
    return input_list, output_list


def preprocess_data_parallel(data_dir: str, min_sentence_num: int = 1, max_sentence_num: int | float = float("inf")):
    input_list = []
    output_list = []
    with open(data_dir, 'r') as json_in:
        data_dict = json.load(json_in)
        conversations = data_dict["text_list"]
        speaker_labels = [speaker_to_ints(speaker_ids) for speaker_ids in data_dict["speaker_list"]]
    num_cores = 8
    chunk_size = len(conversations) // num_cores
    args = [
        (conversations[i:i + chunk_size], speaker_labels[i:i + chunk_size], min_sentence_num, max_sentence_num)
        for i in range(0, len(conversations), chunk_size)
    ]
    with Pool(num_cores) as pool:
        results = pool.map(preprocess_data_chunk, args)
    for res in results:
        input_list.extend(res[0])
        output_list.extend(res[1])
    return input_list, output_list


# Hyper parameters
ENCODER_MAX_LENGTH = 512
DECODER_MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 3
BEGIN_OF_SENTENCE = " <bos> "

if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-3b', cache_dir="./tokenizers", model_max_length=ENCODER_MAX_LENGTH)
    tokenizer.add_special_tokens({"additional_special_tokens": [BEGIN_OF_SENTENCE]})
    tokenizer.save_pretrained(f"./tokenizer_bos")
    model = T5ForConditionalGeneration.from_pretrained('t5-3b', cache_dir="./models")
    model.resize_token_embeddings(len(tokenizer))

    # Create dataset and dataloader
    data_train_dir = "/local/scratch/pwu54/Text-based SD Dataset/dataset7_align_train_sent_2sp.json"
    input_train, output_train = preprocess_data(data_train_dir, 2, 4)
    dataset_train = Dataset.from_dict({"conversations": input_train, "speaker_labels": output_train})
    dataset_train = dataset_train.map(
        process_data_to_model_inputs,
        batched=True,
        # num_proc=8,
        remove_columns=["conversations", "speaker_labels"],
    )

    dataset_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 3. Define Training Arguments and Initialize Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results/t5-3b',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        optim="adafactor",
        learning_rate=1e-4,
        # gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        save_strategy="epoch"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
    )

    # 4. Train the Model
    trainer.train()

    # 5. Inference
    # def predict_speaker_sequence(model, tokenizer, conversation):
    #     input_text = " ".join(conversation)
    #     input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    #     output = model.generate(input_ids)
    #     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(decoded_output)
    #     return list(map(int, decoded_output.split()))
    #
    #
    # conversation_test = ["Hey, are you available?", "Yes, what's up?", "Let's discuss the project."]
    # predicted_sequence = predict_speaker_sequence(model, tokenizer, conversation_test)
    # print(predicted_sequence)

    # Give output from test
    # outputs = model.generate(input_ids=dataset_test['input_ids'], attention_mask=dataset_test['attention_mask'],
    #                          max_length=DECODER_MAX_LENGTH)
    # predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # for i in range(len(predictions)):
    #     print(predictions[i])
