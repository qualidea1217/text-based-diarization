import ast
import json
import os
import random
import string

import openai
import spacy

openai.api_key = "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"  # do not upload this to github
model = "gpt-4-1106-preview"
# model = "gpt-3.5-turbo-1106"

dir_dict = {"AMI audio": "/local/scratch/pwu54/Text-based SD Dataset/AMI/audio/",
            "AMI gt": "/local/scratch/pwu54/Text-based SD Dataset/AMI/transcript/",
            "AMI text": "/local/scratch/pwu54/Text-based SD Dataset/AMI/whisper_output/",
            "CallFriend audio": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/audio/",
            "CallFriend gt": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/transcript/",
            "CallFriend text": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/whisper_output/",
            "CallHome English audio": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/audio/",
            "CallHome English gt": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/transcript/",
            "CallHome English text": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/whisper_output/",
            "CHiME-5 audio": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/audio/",
            "CHiME-5 gt": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/transcript/",
            "CHiME-5 text": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/whisper_output/",
            "DailyTalk audio": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/audio/",
            "DailyTalk gt": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/transcript/",
            "DailyTalk text": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/whisper_output/",
            "ICSI audio": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/audio/",
            "ICSI gt": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/transcript/",
            "ICSI text": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/whisper_output/",
            "SBCSAE audio": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/audio/",
            "SBCSAE gt": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/transcript/",
            "SBCSAE text": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/whisper_output/"}

# Hyper parameters
ENCODER_MAX_LENGTH = 512
DECODER_MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 3
BEGIN_OF_SENTENCE = " <bos> "


def speaker_to_ints(input_list):
    unique_dict = {}
    output_list = []
    for item in input_list:
        if item not in unique_dict:
            unique_dict[item] = len(unique_dict)
        output_list.append(unique_dict[item])
    return output_list


def get_train_val_test_filepath(gt_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of the ratios must equal 1.0")
    random.seed(seed)
    gt_filenames = os.listdir(gt_dir)
    random.shuffle(gt_filenames)
    total = len(gt_filenames)
    train_index = int(train_ratio * total)
    val_index = int((train_ratio + val_ratio) * total)
    train_filepaths = [os.path.join(gt_dir, filename) for filename in gt_filenames[:train_index]]
    val_filepaths = [os.path.join(gt_dir, filename) for filename in gt_filenames[train_index:val_index]]
    test_filepaths = [os.path.join(gt_dir, filename) for filename in gt_filenames[val_index:]]
    return train_filepaths, val_filepaths, test_filepaths


def preprocess_conversation(content, merge: bool = True, segment: bool = True):
    text_list = [utterance[1] for utterance in content if utterance[1] not in string.punctuation and utterance[1] not in string.whitespace]
    speaker_list = [utterance[0] for utterance in content if utterance[1] not in string.punctuation and utterance[1] not in string.whitespace]
    if merge:
        merged_texts = []
        merged_speakers = []
        j = 0
        while j < len(text_list):
            if (j < len(text_list) - 1) and (speaker_list[j] == speaker_list[j + 1]):
                merged_sentence = text_list[j]
                while (j < len(text_list) - 1) and (speaker_list[j] == speaker_list[j + 1]):
                    j += 1
                    merged_sentence += " " + text_list[j]
                merged_texts.append(merged_sentence)
                merged_speakers.append(speaker_list[j])
                j += 1
            else:
                merged_texts.append(text_list[j])
                merged_speakers.append(speaker_list[j])
                j += 1
        text_list = merged_texts
        speaker_list = merged_speakers
    if segment:
        spacy.require_gpu()  # if cupy is installed or spacy with gpu support is installed
        nlp = spacy.load("en_core_web_trf")
        sentences = []
        speaker_ids = []
        for j in range(len(text_list)):
            doc = nlp(text_list[j])
            for sent in doc.sents:
                sentences.append(sent.text)
                speaker_ids.append(speaker_list[j])
        text_list = sentences
        speaker_list = speaker_ids

    text_list_filter = [text_list[i] for i in range(len(text_list)) if text_list[i] not in string.punctuation and text_list[i] not in string.whitespace]
    speaker_list_filter = [speaker_list[i] for i in range(len(speaker_list)) if text_list[i] not in string.punctuation and text_list[i] not in string.whitespace]
    text_list, speaker_list = text_list_filter, speaker_list_filter
    return text_list, speaker_list


def get_conversation_w_speaker(conversation: list[str], speaker_label: list):
    return [[speaker_label[i], conversation[i]] for i in range(len(conversation))]


def get_conversation_metrics_exact(content: list, content_pred: list):
    """If both conversation have exact same words and 2 speaker only,
    both TDER and DF1 falls back to accuracy, which is 1 - WDER"""
    n_incorrect_utterance = 0
    n_total_utterance = 0
    n_incorrect_token = 0
    n_total_token = 0
    for i in range(len(content)):
        n_total_utterance += 1
        n_total_token += len(content[i][1].split())
        if content[i][0] != content_pred[i][0]:
            n_incorrect_utterance += 1
            n_incorrect_token += len(content[i][1].split())
    n_correct_token = n_total_token - n_incorrect_token
    n_correct_utterance = n_total_utterance - n_incorrect_utterance
    df1 = n_correct_token / n_total_token  # also accuracy on token
    tder = n_incorrect_token / n_total_token  # also wder
    acc_utterance = n_correct_utterance / n_total_utterance
    return df1, tder, acc_utterance


def openai_chat(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who knows how to predict speaker labels of conversation."},
            {"role": "user", "content": prompt}
        ]
    )
    output_str = response["choices"][0]["message"]["content"]
    return output_str


def predict_single_input(input_text: str):
    prompt = f"""For the following list of sentences from a 2 speaker conversation, please generate a list of 0 and 1 
    to classify which speaker is speaking which sentence according to the sequence of occurrence of the speaker. 
    Each sentence begins with  {BEGIN_OF_SENTENCE}. 0 for first appeared speaker, 1 for second appeared speaker. 
    Return only python list. Sentences to be predicted: {input_text}"""
    output_list = None
    for _ in range(5):
        try:
            output_str = openai_chat(prompt)
            output_list = ast.literal_eval(output_str)
            break
        except:
            continue
    return output_list


def evaluate_conversation(conversation: list[str], speaker_label: list):
    speaker_label = speaker_to_ints(speaker_label)
    single_input = "".join([BEGIN_OF_SENTENCE + sentence for sentence in conversation])
    speaker_label_pred = predict_single_input(single_input)
    content = get_conversation_w_speaker(conversation, speaker_label)
    content_pred = get_conversation_w_speaker(conversation, speaker_label_pred)
    df1, tder, acc_utterance = get_conversation_metrics_exact(content, content_pred)
    return df1, tder, acc_utterance


if __name__ == "__main__":
    val_filepath_all = []
    test_filepath_all = []
    # for gt_dir in ["AMI gt", "CallFriend gt", "CallHome English gt", "CHiME-5 gt", "DailyTalk gt", "ICSI gt",
    #                "SBCSAE gt"]:
    for gt_dir in ["DailyTalk gt"]:
        gt_dir = dir_dict[gt_dir].replace("transcript", "whisper_align")
        train_filepaths, val_filepaths, test_filepaths = get_train_val_test_filepath(gt_dir)
        val_filepath_all.extend(val_filepaths)
        test_filepath_all.extend(test_filepaths)

    tder_list = []
    df1_list = []
    acc_u_list = []
    for filepath in test_filepath_all:
        if os.path.splitext(filepath)[1] == ".json":
            with open(filepath, 'r') as json_in:
                content = json.load(json_in)  # fill in the content of conversation in this format
        conversation, speaker_label = preprocess_conversation(content)
        if len(set(speaker_label)) != 2:
            continue
        df1, tder, acc_u = evaluate_conversation(conversation, speaker_label)
        print(f"filepath: {filepath}\nDF1: {df1}, TDER: {tder}, ACC_U: {acc_u}")
        tder_list.append(tder)
        df1_list.append(df1)
        acc_u_list.append(acc_u)
    print(f"avg DF1: {sum(df1_list) / len(df1_list)}, avg TDER: {sum(tder_list) / len(tder_list)}, avg acc utterance: {sum(acc_u_list) / len(acc_u_list)}")
