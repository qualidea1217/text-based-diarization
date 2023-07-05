import json
import os
import re
import string
import wave
import xml.etree.ElementTree


def daily_talk(daily_talk_dir: str):
    for sub_dir in os.listdir(daily_talk_dir):
        sub_dir = os.path.join(daily_talk_dir, sub_dir)

        def sort_key(sub_file):
            basename = os.path.basename(sub_file)  # Get the filename without directory.
            base, _ = os.path.splitext(basename)  # Get the filename without extension.
            x, _, _ = base.split('_')  # Split the filename at the underscores.
            return int(x)  # Convert the 'x' part to an integer and return it.

        # remove files so that there will not be naming collision and make sure collecting all wav and txt files works
        os.remove(os.path.join(sub_dir, os.path.basename(sub_dir) + ".wav"))
        os.remove(os.path.join(sub_dir, os.path.basename(sub_dir) + ".json"))

        # preprocess audio
        wav_files = [os.path.join(sub_dir, sub_file) for sub_file in os.listdir(sub_dir) if sub_file.endswith(".wav")]
        wav_files.sort(key=sort_key)
        with wave.open(os.path.join(sub_dir, os.path.basename(sub_dir) + ".wav"), 'wb') as wav_out:
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
        with open(os.path.join(sub_dir, os.path.basename(sub_dir) + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def icsi(icsi_dir: str):
    for mrt_file in os.listdir(icsi_dir):
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
        with open(os.path.join(icsi_dir, os.path.basename(mrt_file) + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def sbcsae(trn_dir: str, cha_dir: str):
    # extract speaker name
    all_speakers = []
    for cha_file in os.listdir(cha_dir):
        file_speakers = []
        with open(os.path.join(cha_dir, cha_file), 'r', encoding="utf-8", errors="ignore") as cha:
            for line in cha.readlines():
                if line.startswith("@ID:"):
                    parts = line.split("|")
                    if "Speaker" in parts:
                        file_speakers.append(parts[2])
        print(file_speakers)
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
        with open(os.path.join(trn_dir, os.path.basename(trn_file) + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def callhome(cha_dir: str):
    TRANS = str.maketrans('', '', string.punctuation)
    for cha_file in os.listdir(cha_dir):
        transcript = []
        speaker = ""
        utterance = []
        with open(os.path.join(cha_dir, cha_file), 'r', encoding="utf-8", errors="ignore") as cha:
            for line in cha.readlines():
                line = line.strip().split()
                if line[0].startswith('*'):
                    # add to transcript
                    if speaker != "" and len(utterance) != 0:
                        utterance = [s.translate(TRANS) for s in utterance if s[:2] != "&=" and '\x15' not in s and s.translate(TRANS) != '']
                        transcript.append([speaker, ' '.join(utterance)])
                        utterance = []
                    speaker = line[0].translate(TRANS)
                    utterance.extend(line[1:])
                else:
                    if line[0][0] != '%' and line[0][0] != '@':
                        utterance.extend(line)
            # add final utterance to transcript
            if speaker != "" and len(utterance) != 0:
                utterance = [s.translate(TRANS) for s in utterance if s[:2] != "&=" and '\x15' not in s and s.translate(TRANS) != '']
                transcript.append([speaker, ' '.join(utterance)])
        with open(os.path.join(cha_dir, os.path.basename(cha_file) + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


def libricss(libricss_dir: str):
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


def chime5(chime5_dir: str):
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
                with open(os.path.join(root, os.path.splitext(os.path.basename(filename))[0]) + "_convert.json", 'w') as json_out:
                    json.dump(transcript, json_out, indent=4)


def ami(ami_dir: str):

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
            with open(os.path.join(ami_dir, current_meeting + ".json"), 'w') as json_out:
                json.dump(transcript, json_out, indent=4)
            word_list = []
        word_list.extend([(float(e.attrib['starttime']), speaker, e.text) for e in word_elements if "starttime" in e.attrib])
        current_meeting = meeting
    if len(word_list) != 0:
        word_list.sort(key=lambda x: x[0])
        transcript = group_to_utterance(word_list)
        # write to json
        with open(os.path.join(ami_dir, current_meeting + ".json"), 'w') as json_out:
            json.dump(transcript, json_out, indent=4)


if __name__ == "__main__":
    # daily_talk("D:\\Text-based SD Dataset\\DailyTalk\\data")
    # icsi("D:\\Text-based SD Dataset\\ICSI\\ICSI_original_transcripts\\transcripts")
    # sbcsae("D:\\Text-based SD Dataset\\Santa Barbara Corpus of Spoken American English\\SBCorpus\\TRN",
    #        "D:\\Text-based SD Dataset\\Santa Barbara Corpus of Spoken American English\\SBCSAE_chat\\SBCSAE")
    # callhome("D:\\Text-based SD Dataset\\CallHome English\\eng\\eng")
    # libricss("D:\\Text-based SD Dataset\\LibriCSS\\for_release")
    # chime5("D:\\Text-based SD Dataset\\CHiME-5\\CHiME5_transcriptions\\CHiME5\\transcriptions")
    ami("D:\\Text-based SD Dataset\\AMI\\ami_public_manual_1.6.2\\words")
