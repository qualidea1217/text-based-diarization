import csv
import json
import os

from align4d import align


def form_utterance(align_result: dict, remove_empty_utterance: bool = True) -> tuple[list, list]:
    tokens = align_result['hypothesis']
    references = align_result['reference']
    utterances = []
    speakers = []
    current_speaker = None
    current_utterance = []
    for i, token in enumerate(tokens):
        for speaker, ref_tokens in references.items():
            if ref_tokens[i]:
                if current_speaker == speaker:
                    current_utterance.append(token)
                else:
                    if current_utterance:
                        utterances.append(" ".join(current_utterance).strip())
                        speakers.append(current_speaker)
                    current_speaker = speaker
                    current_utterance = [token]
                break
    if current_utterance:
        utterances.append(" ".join(current_utterance).strip())
        speakers.append(current_speaker)
    if remove_empty_utterance:
        utterances_filter = []
        speakers_filter = []
        for i in range(len(utterances)):
            if utterances[i].strip() != "":
                utterances_filter.append(utterances[i])
                speakers_filter.append(speakers[i])
        return utterances_filter, speakers_filter
    return utterances, speakers


def align_whisper_to_gt(whisper_output_file: str, gt_file: str):
    with open(whisper_output_file, 'r') as hypo_in:
        hypothesis = hypo_in.read()
    with open(gt_file, 'r') as ref_in:
        reference = json.load(ref_in)
    align_result = align.align(hypothesis, reference)
    return align_result


def write_align_result_to_csv(align_result: dict, file_name: str):
    with open(file_name, 'w') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(align_result["hypothesis"])
        for key, value in align_result["reference"].items():
            writer.writerow(value)


if __name__ == "__main__":
    data = {
        "hypothesis": ['ok', 'I', 'am', 'a', 'fish.', 'Are', 'you?', 'Hello', 'there.', 'How', 'are', 'you?', 'ok'],
        "reference": {
            "A": ['', 'I', 'am', 'a', 'fish.', '', '', '', '', '', '', '', ''],
            "B": ['okay.', '', '', '', '', '', '', '', '', '', '', '', ''],
            "C": ['', '', '', '', '', 'Are', 'you?', '', '', '', '', '', ''],
            "D": ['', '', '', '', '', '', '', 'Hello', 'there.', '', '', '', ''],
            "E": ['', '', '', '', '', '', '', '', '', 'How', 'are', 'you?', ''],
        }
    }
    utterance, speakers = form_utterance(data)
    print(utterance)
    print(speakers)

