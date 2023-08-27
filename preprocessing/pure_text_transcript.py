import json
import time

import pandas as pd
import spacy


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result

    return wrapper


@timer
def get_interview_dataset(csv_dir: str, segmentation: None | str = None, speaker_id: None | str = None):
    # extract episode number, utterances, and speaker ids to python lists
    df = pd.read_csv(csv_dir)
    df = df.sort_values(by=["episode", "episode_order"])
    episode_list = []
    text_list = []
    speaker_list = []
    for episode in df['episode'].unique():
        utterances = []
        speaker_ids = []
        df_episode = df[df["episode"] == episode].sort_values(by="episode_order")
        for _, row in df_episode.iterrows():
            if type(row["utterance"]) != float:  # some utterance are nan
            # if type(row["utterance"]) != float and row["speaker"] != "_NO_SPEAKER": # remove no-speaker utterance
                utterances.append(row["utterance"])
                speaker_ids.append(row["speaker"])
        episode_list.append(int(episode))
        text_list.append(utterances)
        speaker_list.append(speaker_ids)

    # for each utterance in each episode, do sentence segmentation and adjust speaker id accordingly
    # use spacy to do sentence segmentation
    if segmentation == "sentence":
        spacy.require_gpu()  # if cupy is installed or spacy with gpu support is installed
        nlp = spacy.load("en_core_web_trf")
        for i in range(len(text_list)):
            sentences = []
            speaker_ids = []
            for j in range(len(text_list[i])):
                doc = nlp(text_list[i][j])
                for sent in doc.sents:
                    sentences.append(sent.text)
                    speaker_ids.append(speaker_list[i][j])
            text_list[i] = sentences
            speaker_list[i] = speaker_ids
    return episode_list, text_list, speaker_list


if __name__ == "__main__":
    episode_list, text_list, speaker_list = get_interview_dataset("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/utterances.csv")
    dict_out = {"episode_list": episode_list, "text_list": text_list, "speaker_list": speaker_list}
    with open("interview_utterance.json", 'w') as json_out:
        json.dump(dict_out, json_out, indent=4)
