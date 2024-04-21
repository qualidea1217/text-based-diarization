import copy

import numpy as np
from scipy.optimize import linear_sum_assignment


def get_global_label(speaker_label_pred: list, max_speaker: int | None = None) -> list:
    speaker_label_pred_global = copy.deepcopy(speaker_label_pred)
    for i in range(len(speaker_label_pred_global) - 1):
        if speaker_label_pred_global[i][0][0] == 0 and speaker_label_pred_global[i + 1][0][0] == 0:
            sentence_range, label = next((e for e in reversed(speaker_label_pred_global) if e[0][0] == 0))
            sentence_range_next, label_next = speaker_label_pred_global[i]
        else:
            sentence_range, label = speaker_label_pred_global[i]
            sentence_range_next, label_next = speaker_label_pred_global[i + 1]
        intersection = (max(sentence_range[0], sentence_range_next[0]), min(sentence_range[1], sentence_range_next[1]))
        label_index_range = (intersection[0] - sentence_range[0], intersection[1] - sentence_range[0])
        label_next_index_range = (intersection[0] - sentence_range_next[0], intersection[1] - sentence_range_next[0])
        label_map = find_label_mapping_int(label[label_index_range[0]:label_index_range[1]],
                                           label_next[label_next_index_range[0]:label_next_index_range[1]])
        label_next_global = [label_map.get(ln, '*') for ln in label_next]

        # Handle new speaker that is not in the mapping (new speaker in the end)
        for j in range(len(label_next)):
            if label_next_global[j] == '*':
                if max_speaker is None:
                    label_next_global[j] = max([e for e in label_next_global if e != '*']) + 1
                else:
                    label_next_global[j] = (max([e for e in label_next_global if e != '*']) + 1) % max_speaker

        if speaker_label_pred_global[i][0][0] == 0 and speaker_label_pred_global[i + 1][0][0] == 0:
            speaker_label_pred_global[i] = (sentence_range_next, label_next_global)
        else:
            speaker_label_pred_global[i + 1] = (sentence_range_next, label_next_global)
    return speaker_label_pred_global


def find_label_mapping_int(ref_labels: list[int], new_labels: list[int], sentence_lengths: list[int] = None) -> dict:
    # Use the previously defined function to find the label mapping for the new lists
    # Adjusting the function to work with integer labels within the execution cell for direct execution
    # potential improvement: add length of sentence as a weight
    if len(ref_labels) != len(new_labels):
        raise ValueError("Reference and new labels must have the same length.")
    if sentence_lengths is None:
        sentence_lengths = [1] * len(new_labels)  # default weight as 1 (no weight)
    unique_ref_labels = sorted(set(ref_labels))
    unique_new_labels = sorted(set(new_labels))
    ref_label_to_index = {label: index for index, label in enumerate(unique_ref_labels)}
    new_label_to_index = {label: index for index, label in enumerate(unique_new_labels)}
    confusion_matrix = np.zeros((len(unique_ref_labels), len(unique_new_labels)), dtype=int)
    for ref_label, new_label, sentence_length in zip(ref_labels, new_labels, sentence_lengths):
        ref_index = ref_label_to_index[ref_label]
        new_index = new_label_to_index[new_label]
        confusion_matrix[ref_index, new_index] += sentence_length
    cost_matrix = confusion_matrix.max() - confusion_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {unique_new_labels[new]: unique_ref_labels[ref] for ref, new in zip(row_ind, col_ind)}
    return mapping


def aggregate(speaker_label_pred: list, conversation_length: int) -> list:
    aggregation_log = [{} for _ in range(conversation_length)]
    for sentence_range, label in speaker_label_pred:
        for i in range(sentence_range[0], sentence_range[1]):
            sl = label[i - sentence_range[0]]
            aggregation_log[i][sl] = aggregation_log[i].get(sl, 0) + 1  # increment by 1
    print(aggregation_log)
    aggregation_result = [max(d, key=lambda k: d[k]) for d in aggregation_log]
    return aggregation_result


if __name__ == '__main__':
    min_sentence_num = 2
    max_sentence_num = 6
    conversation = [0, 2, 1, 3, 0, 2, 3, 1, 0, 3]
    # speaker_label_pred = []
    # for i in range(min_sentence_num, len(conversation) + max_sentence_num - min_sentence_num + 1):
    #     begin = i - max_sentence_num if i - max_sentence_num >= 0 else 0
    #     end = i if i <= len(conversation) else len(conversation)
    #     speaker_label_pred.append(((begin, end), conversation[begin:end]))
    speaker_label_pred = [
        ((0, 2), [0, 1]),
        ((0, 3), [0, 1, 2]),
        ((0, 4), [0, 1, 2, 3]),
        ((0, 5), [0, 1, 2, 3, 0]),
        ((0, 6), [0, 1, 2, 3, 0, 1]),
        ((1, 7), [0, 1, 2, 3, 0, 2]),
        ((2, 8), [0, 1, 2, 3, 1, 0]),
        ((3, 9), [0, 1, 2, 0, 3, 1]),
        ((4, 10), [0, 1, 2, 3, 0, 2]),
        ((5, 10), [0, 1, 2, 3, 1]),
        ((6, 10), [0, 1, 2, 0]),
        ((7, 10), [0, 1, 2]),
        ((8, 10), [0, 1])
    ]
    speaker_label_pred_global = get_global_label(speaker_label_pred)
    print(speaker_label_pred_global)
    aggregation_result = aggregate(speaker_label_pred_global, len(conversation))
    print(aggregation_result)
    final_mapping = find_label_mapping_int(conversation, aggregation_result)
    final_result = [final_mapping[l] for l in aggregation_result]
    print(final_result)


