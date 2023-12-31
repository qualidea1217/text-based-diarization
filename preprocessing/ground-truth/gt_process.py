import json
import os
import re
import random
import string
import time
import wave
import xml.etree.ElementTree

import pandas as pd
import spacy

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


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper


def daily_talk(root_dir: str):
    daily_talk_dir = os.path.join(root_dir, "data")
    for sub_dir in os.listdir(daily_talk_dir):
        sub_dir = os.path.join(daily_talk_dir, sub_dir)

        def sort_key(sub_file):
            basename = os.path.basename(sub_file)  # Get the filename without directory.
            base, _ = os.path.splitext(basename)  # Get the filename without extension.
            x, _, _ = base.split('_')  # Split the filename at the underscores.
            return int(x)  # Convert the 'x' part to an integer and return it.

        # preprocess audio
        wav_files = [os.path.join(sub_dir, sub_file) for sub_file in os.listdir(sub_dir) if sub_file.endswith(".wav")]
        wav_files.sort(key=sort_key)
        if not os.path.exists(os.path.join(root_dir, "audio")):
            os.makedirs(os.path.join(root_dir, "audio"))
        with wave.open(os.path.join(root_dir, "audio", os.path.basename(sub_dir) + ".wav"), 'wb') as wav_out:
            for wav_file in wav_files:
                with wave.open(wav_file, 'rb') as wav_in:
                    if not wav_out.getnframes():
                        wav_out.setparams(wav_in.getparams())
                    wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

        # preprocess text
        txt_files = [os.path.join(sub_dir, sub_file) for sub_file in os.listdir(sub_dir) if sub_file.endswith(".txt")]
        txt_files.sort(key=sort_key)
        transcript = []
        for txt_file in txt_files:
            base, _ = os.path.splitext(os.path.basename(txt_file))  # Get the filename without extension.
            _, speaker, _ = base.split('_')  # Split the filename at the underscores.
            with open(txt_file, 'r', encoding="utf-8") as txt_in:
                transcript.append([speaker, txt_in.read()])
        if not os.path.exists(os.path.join(root_dir, "transcript")):
            os.makedirs(os.path.join(root_dir, "transcript"))
        with open(os.path.join(root_dir, "transcript", os.path.basename(sub_dir) + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def icsi(root_dir: str):
    icsi_dir = os.path.join(root_dir, "ICSI_original_transcripts", "transcripts")
    for mrt_file in os.listdir(icsi_dir):
        if mrt_file == "preambles.mrt" or mrt_file == "readme":
            continue
        mrt_file = os.path.join(icsi_dir, mrt_file)
        tree = xml.etree.ElementTree.parse(mrt_file)
        root = tree.getroot()
        # Find all 'Segment' elements
        segments = root.findall('.//Segment')
        transcript = []
        for segment in segments:
            speaker = segment.get('Participant')
            utterance = segment.text.strip()
            if utterance != '':
                transcript.append([speaker, utterance])
        if not os.path.exists(os.path.join(root_dir, "transcript")):
            os.makedirs(os.path.join(root_dir, "transcript"))
        with open(os.path.join(root_dir, "transcript", os.path.splitext(os.path.basename(mrt_file))[0] + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def sbcsae(root_dir: str):
    cha_dir = os.path.join(root_dir, "SBCSAE_chat", "SBCSAE")
    trn_dir = os.path.join(root_dir, "SBCorpus", "TRN")
    # extract speaker name
    all_speakers = []
    for cha_file in os.listdir(cha_dir):
        if os.path.splitext(cha_file)[1] != ".cha":
            continue
        file_speakers = []
        with open(os.path.join(cha_dir, cha_file), 'r', encoding="utf-8", errors="ignore") as cha:
            for line in cha.readlines():
                if line.startswith("@ID:"):
                    parts = line.split("|")
                    if "Speaker" in parts:
                        file_speakers.append(parts[2])
        all_speakers.append(file_speakers)

    TRANS = str.maketrans('', '', string.punctuation)
    for i, trn_file in enumerate(os.listdir(trn_dir)):
        transcript = []
        with open(os.path.join(trn_dir, trn_file), 'r', encoding="utf-8", errors="ignore") as trn:
            speaker = ""
            utterance = []
            for line in trn.readlines():
                line = [s.translate(TRANS) for s in line.strip().split() if s.translate(TRANS) != '']
                if len(line) > 2:
                    if any(s == line[2] or s == line[2][:4] for s in all_speakers[i]):
                        if speaker != "" and len(utterance) != 0:
                            transcript.append([speaker, ' '.join(utterance)])
                            utterance = []
                        speaker = line[2]
                        utterance.extend(line[3:])
                    else:
                        utterance.extend(line[2:])
            if speaker != "" and len(utterance) != 0:
                transcript.append([speaker, ' '.join(utterance)])
        if not os.path.exists(os.path.join(root_dir, "transcript")):
            os.makedirs(os.path.join(root_dir, "transcript"))
        with open(os.path.join(root_dir, "transcript", os.path.splitext(os.path.basename(trn_file))[0] + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def callhome(root_dir: str):
    cha_dir = os.path.join(root_dir, "eng", "eng")
    TRANS = str.maketrans('', '', string.punctuation)
    for cha_file in os.listdir(cha_dir):
        if os.path.splitext(cha_file)[1] != ".cha":
            continue
        transcript = []
        speaker = ""
        utterance = []
        begin = False
        with open(os.path.join(cha_dir, cha_file), 'r', encoding="utf-8", errors="ignore") as cha:
            for line in cha.readlines():
                line = line.strip().split()
                if line[0].startswith('*'):
                    # add to transcript
                    begin = True
                    if speaker != "" and len(utterance) != 0:
                        utterance = [s.translate(TRANS) for s in utterance if s[:2] != "&=" and '\x15' not in s and s.translate(TRANS) != '']
                        transcript.append([speaker, ' '.join(utterance)])
                        utterance = []
                    speaker = line[0].translate(TRANS)
                    utterance.extend(line[1:])
                else:
                    if line[0][0] != '%' and line[0][0] != '@' and begin:
                        utterance.extend(line)
            # add final utterance to transcript
            if speaker != "" and len(utterance) != 0:
                utterance = [s.translate(TRANS) for s in utterance if s[:2] != "&=" and '\x15' not in s and s.translate(TRANS) != '']
                transcript.append([speaker, ' '.join(utterance)])
        if not os.path.exists(os.path.join(root_dir, "transcript")):
            os.makedirs(os.path.join(root_dir, "transcript"))
        with open(os.path.join(root_dir, "transcript", os.path.splitext(os.path.basename(cha_file))[0] + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def libricss(root_dir: str):
    libricss_dir = os.path.join(root_dir, "for_release")
    for root, dirnames, filenames in os.walk(libricss_dir):
        if len(filenames) == 1 and filenames[0] == "meeting_info.txt":
            txt_path = os.path.join(root, filenames[0])
            transcript = []
            with open(txt_path, 'r', encoding="utf-8", errors="ignore") as txt:
                for line in txt.readlines()[1:]:
                    line = line.strip().split()
                    speaker = line[2]
                    utterance = [s.lower() for s in line[4:]]
                    transcript.append([speaker, ' '.join(utterance)])
            with open(os.path.join(root, os.path.basename(filenames[0]) + ".json"), 'w') as json_out:
                json.dump(transcript, json_out, indent=4)


def chime5(root_dir: str):
    chime5_dir = os.path.join(root_dir, "CHiME5_transcriptions", "CHiME5", "transcriptions")
    for root, dirnames, filenames in os.walk(chime5_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == ".json":
                transcript = []
                with open(os.path.join(root, filename), 'r', encoding="utf-8", errors="ignore") as json_in:
                    data = json.load(json_in)
                    for record in data:
                        try:
                            speaker = record["speaker"]
                            utterance = re.sub(r'\[.*?\]', '', record["words"].strip())
                            transcript.append([speaker, utterance])
                        except KeyError:
                            pass
                if not os.path.exists(os.path.join(root_dir, "transcript")):
                    os.makedirs(os.path.join(root_dir, "transcript"))
                with open(os.path.join(root_dir, "transcript", os.path.splitext(os.path.basename(filename))[0]) + ".json", 'w') as json_out:
                    json.dump(transcript, json_out, indent=4)


def ami(root_dir: str):
    ami_dir = os.path.join(root_dir, "ami_public_manual_1.6.2", "words")

    def group_to_utterance(word_list: list[tuple]) -> list:
        transcript = []
        current_speaker = None
        utterance = []
        for word in word_list:
            speaker = word[1]
            if speaker == current_speaker:
                # If the speaker hasn't changed, continue accumulating the words
                utterance.append(word[2])
            else:
                # If the speaker has changed, finish the current utterance and start a new one
                if current_speaker is not None:
                    transcript.append([current_speaker, ' '.join(utterance)])
                current_speaker = speaker
                utterance = [word[2]]
            # Add the last utterance
        if utterance:
            transcript.append([current_speaker, ' '.join(utterance)])
        return transcript

    current_meeting = None
    word_list = []
    for xml_file in os.listdir(ami_dir):
        meeting, speaker = xml_file.split('.')[:2]
        xml_file = os.path.join(ami_dir, xml_file)
        tree = xml.etree.ElementTree.parse(xml_file)
        root = tree.getroot()
        word_elements = root.findall(".//w")
        if meeting != current_meeting and current_meeting is not None:
            word_list.sort(key=lambda x: x[0])
            transcript = group_to_utterance(word_list)
            # write to json
            if not os.path.exists(os.path.join(root_dir, "transcript")):
                os.makedirs(os.path.join(root_dir, "transcript"))
            with open(os.path.join(root_dir, "transcript", current_meeting + ".json"), 'w') as json_out:
                json.dump(transcript, json_out, indent=4)
            word_list = []
        word_list.extend([(float(e.attrib['starttime']), speaker, e.text) for e in word_elements if "starttime" in e.attrib])
        current_meeting = meeting
    if len(word_list) != 0:
        word_list.sort(key=lambda x: x[0])
        transcript = group_to_utterance(word_list)
        # write to json
        if not os.path.exists(os.path.join(root_dir, "transcript")):
            os.makedirs(os.path.join(root_dir, "transcript"))
        with open(os.path.join(root_dir, "transcript", current_meeting + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def callfriend(root_dir: str):
    cha_dir = os.path.join(root_dir, "eng", "eng")
    TRANS = str.maketrans('', '', string.punctuation)
    for cha_file in os.listdir(cha_dir):
        if os.path.splitext(cha_file)[1] != ".cha":
            continue
        transcript = []
        speaker = ""
        utterance = []
        begin = False
        with open(os.path.join(cha_dir, cha_file), 'r', encoding="utf-8", errors="ignore") as cha:
            for line in cha.readlines():
                line = line.strip().split()
                if line[0].startswith('*'):
                    # add to transcript
                    begin = True
                    if speaker != "" and len(utterance) != 0:
                        utterance = [s.translate(TRANS) for s in utterance if s[:2] != "&=" and '\x15' not in s and s.translate(TRANS) != '']
                        transcript.append([speaker, ' '.join(utterance).encode("ascii", "ignore").decode("ascii")])
                        utterance = []
                    speaker = line[0].translate(TRANS)
                    utterance.extend(line[1:])
                else:
                    if line[0][0] != '%' and line[0][0] != '@' and begin:
                        utterance.extend(line)
            # add final utterance to transcript
            if speaker != "" and len(utterance) != 0:
                utterance = [s.translate(TRANS) for s in utterance if s[:2] != "&=" and '\x15' not in s and s.translate(TRANS) != '']
                transcript.append([speaker, ' '.join(utterance).encode("ascii", "ignore").decode("ascii")])
        if not os.path.exists(os.path.join(root_dir, "transcript")):
            os.makedirs(os.path.join(root_dir, "transcript"))
        with open(os.path.join(root_dir, "transcript", os.path.splitext(os.path.basename(cha_file))[0] + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def interview(csv_dir: str, segmentation: None | str = None, max_speaker: None | int = 2):
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

    if max_speaker:
        episode_list = [episode_list[i] for i in range(len(episode_list)) if len(set(speaker_list[i])) <= max_speaker]
        text_list = [text_list[i] for i in range(len(text_list)) if len(set(speaker_list[i])) <= max_speaker]
        speaker_list = [speaker_list[i] for i in range(len(speaker_list)) if len(set(speaker_list[i])) <= max_speaker]

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

    dict_out = {"episode_list": episode_list, "text_list": text_list, "speaker_list": speaker_list}
    with open("interview_sentence_2sp.json", 'w') as json_out:
        json.dump(dict_out, json_out, indent=4)


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


def get_gt_scd_data(filepaths: list[str], output_json_name: str, merge: bool = True, segment: bool = True, max_speaker: None | int = None):
    text_list = []
    speaker_list = []
    for filepath in filepaths:
        if os.path.splitext(filepath)[1] == ".json":
            with open(filepath, 'r') as json_in:
                content = json.load(json_in)
                text_list = [utterance[1] for utterance in content if
                             utterance[1] not in string.punctuation and utterance[1] not in string.whitespace]
                speaker_list = [utterance[0] for utterance in content if
                                utterance[1] not in string.punctuation and utterance[1] not in string.whitespace]
    if max_speaker:
        text_list = [text_list[i] for i in range(len(text_list)) if len(set(speaker_list[i])) <= max_speaker]
        speaker_list = [speaker_list[i] for i in range(len(speaker_list)) if len(set(speaker_list[i])) <= max_speaker]
    if merge:
        for i in range(len(text_list)):
            merged_texts = []
            merged_speakers = []
            j = 0
            while j < len(text_list[i]):
                if (j < len(text_list[i]) - 1) and (speaker_list[i][j] == speaker_list[i][j + 1]):
                    merged_sentence = text_list[i][j]
                    while (j < len(text_list[i]) - 1) and (speaker_list[i][j] == speaker_list[i][j + 1]):
                        j += 1
                        merged_sentence += " " + text_list[i][j]
                    merged_texts.append(merged_sentence)
                    merged_speakers.append(speaker_list[i][j])
                    j += 1
                else:
                    merged_texts.append(text_list[i][j])
                    merged_speakers.append(speaker_list[i][j])
                    j += 1
            text_list[i] = merged_texts
            speaker_list[i] = merged_speakers

    if segment:
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

    for i in range(len(text_list)):
        text_list_filter = []
        speaker_list_filter = []
        for j in range(len(text_list[i])):
            if len(text_list[i][j]) > 1:
                text_list_filter.append(text_list[i][j])
                speaker_list_filter.append(speaker_list[i][j])
        text_list[i] = text_list_filter
        speaker_list[i] = speaker_list_filter

    text_list_filter = [text_list[i] for i in range(len(text_list)) if
                        text_list[i] not in string.punctuation and text_list[i] not in string.whitespace]
    speaker_list_filter = [speaker_list[i] for i in range(len(speaker_list)) if
                           text_list[i] not in string.punctuation and text_list[i] not in string.whitespace]
    text_list, speaker_list = text_list_filter, speaker_list_filter

    with open(output_json_name, 'w') as json_out:
        json.dump({"text_list": text_list, "speaker_list": speaker_list}, json_out, indent=4)


if __name__ == "__main__":
    # train_filepath_all = []
    # val_filepath_all = []
    # test_filepath_all = []
    # for gt_dir in ["AMI gt", "CallFriend gt", "CallHome English gt", "CHiME-5 gt", "DailyTalk gt", "ICSI gt", "SBCSAE gt"]:
    #     gt_dir = dir_dict[gt_dir].replace("transcript", "whisper_align")
    #     train_filepaths, val_filepaths, test_filepaths = get_train_val_test_filepath(gt_dir)
    #     train_filepath_all.extend(train_filepaths)
    #     val_filepath_all.extend(val_filepaths)
    #     test_filepath_all.extend(test_filepaths)
    # print("train data")
    # print(train_filepath_all)
    # print("val data")
    # print(val_filepath_all)
    # print("test data")
    # print(test_filepath_all)
    # get_gt_scd_data(train_filepath_all, "dataset7_align_train_sent_2sp.json", max_speaker=2)
    # get_gt_scd_data(val_filepath_all, "dataset7_align_val_sent_2sp.json", max_speaker=2)
    # get_gt_scd_data(test_filepath_all, "dataset7_align_test_sent_2sp.json", max_speaker=2)

    # interview("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/utterances.csv", "sentence", 2)

    with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence.json", 'r') as json_in:
        data_dict = json.load(json_in)
        episode_list = data_dict["episode_list"]
        text_list = data_dict["text_list"]
        speaker_list = data_dict["speaker_list"]

    max_speaker = 2
    episode_list = [episode_list[i] for i in range(len(episode_list)) if len(set(speaker_list[i])) <= max_speaker]
    text_list = [text_list[i] for i in range(len(text_list)) if len(set(speaker_list[i])) <= max_speaker]
    speaker_list = [speaker_list[i] for i in range(len(speaker_list)) if len(set(speaker_list[i])) <= max_speaker]
    print(len(episode_list))
    print(len(text_list))
    print(len(speaker_list))

    dict_out = {"episode_list": episode_list, "text_list": text_list, "speaker_list": speaker_list}
    with open("/local/scratch/pwu54/Text-based SD Dataset/INTERVIEW/interview_sentence_2sp.json", 'w') as json_out:
        json.dump(dict_out, json_out, indent=4)


