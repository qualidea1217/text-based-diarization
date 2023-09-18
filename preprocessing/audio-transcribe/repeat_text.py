import json
import os
import time
from multiprocessing import Pool

from numba import njit

dir_dict = {"AMI audio": "/local/scratch/pwu54/Text-based SD Dataset/AMI/audio/",
            "AMI text": "/local/scratch/pwu54/Text-based SD Dataset/AMI/whisper_output/",
            "CallFriend audio": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/audio/",
            "CallFriend text": "/local/scratch/pwu54/Text-based SD Dataset/CallFriend/whisper_output/",
            "CallHome English audio": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/CallHome/",
            "CallHome English text": "/local/scratch/pwu54/Text-based SD Dataset/CallHome English/whisper_output/",
            "CHiME-5 audio1": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/audio/",
            "CHiME-5 text1": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/whisper_output/",
            "CHiME-5 audio2": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/audio2/",
            "CHiME-5 text2": "/local/scratch/pwu54/Text-based SD Dataset/CHiME-5/whisper_output2/",
            "DailyTalk audio": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/audio/",
            "DailyTalk text": "/local/scratch/pwu54/Text-based SD Dataset/DailyTalk/whisper_output/",
            "ICSI audio": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/Signals/",
            "ICSI text": "/local/scratch/pwu54/Text-based SD Dataset/ICSI/whisper_output/",
            "SBCSAE audio": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/",
            "SBCSAE text": "/local/scratch/pwu54/Text-based SD Dataset/SBCSAE/whisper_output/"}


@njit()
def is_repeating(input_str: str, min_length: int, min_repeat: int) -> bool:
    """
    Check if a string contains substring that is repeating itself.
    :param input_str: String need to check.
    :param min_length: Minimum length of the repeating substring.
    :param min_repeat: Minimum repeating times of the repeating substring.
    :return: boolean.
    """
    max_length = len(input_str) // min_repeat
    for length in range(max_length, min_length - 1, -1):
        for start in range(len(input_str) - length * min_repeat + 1):
            substring = input_str[start:start + length]
            pattern = substring * min_repeat
            if pattern in input_str:
                print(pattern)
                return True
    return False


def is_repeating_file(txt_filename: str, min_length: int, min_repeat: int) -> bool:
    with open(txt_filename, 'r', encoding="utf-8", errors="ignore") as txt:
        return is_repeating(txt.read(), min_length, min_repeat)


def is_repeating_file_parallel(args: tuple) -> tuple | None:
    st = time.time()
    txt_root, txt_filename, min_length, min_repeat = args
    if is_repeating_file(os.path.join(txt_root, txt_filename), min_length, min_repeat):
        et = time.time()
        print(f"file: {txt_filename}, time: {et - st}")
        return os.path.splitext(txt_filename)[0], os.path.join(txt_root, txt_filename)
    et = time.time()
    print(f"file: {txt_filename}, time: {et - st}")
    return None


def get_txt_retry_dict(text_output_dir: str, min_length: int, min_repeat: int, n_process: int = 4):
    """
    Get information of text files need to retry.
    This function uses processpool from multiprocessing to speed up processing on the file level, which uses CPU only.
    :param text_output_dir: Parent directory of all text output files.
    :param min_length: Minimum length that a sequence of text can be recognized to be repeated.
    :param min_repeat: Minimum times of repetition for a sequence of text.
    :param n_process: Number of precesses it used in detecting repetition for multiple files.
    :return txt_retry_dict: Dictionary with information of text files need to retry key: basename, value: path(root + filename)
    """
    args_list = []
    for txt_root, txt_dirnames, txt_filenames in os.walk(text_output_dir):
        for txt_filename in txt_filenames:
            args_list.append((txt_root, txt_filename, min_length, min_repeat))

    with Pool(n_process) as pool:
        results = pool.map(is_repeating_file_parallel, args_list)

    txt_retry_dict = {}  # key: basename, value: path(root + filename)
    for result in results:
        if result is not None:
            basename, path = result
            txt_retry_dict[basename] = path

    return txt_retry_dict


if __name__ == "__main__":
    txt_retry_dict = get_txt_retry_dict(dir_dict["AMI text"], 5, 5, 32)
    with open("txt_retry_ami.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)

    txt_retry_dict = get_txt_retry_dict(dir_dict["CallFriend text"], 5, 5, 32)
    with open("txt_retry_callfriend.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)

    txt_retry_dict = get_txt_retry_dict(dir_dict["CallHome English text"], 5, 5, 32)
    with open("txt_retry_callhome-english.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)

    txt_retry_dict = get_txt_retry_dict(dir_dict["DailyTalk text"], 5, 5, 32)
    with open("txt_retry_dailytalk.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)

    txt_retry_dict = get_txt_retry_dict(dir_dict["ICSI text"], 5, 5, 32)
    with open("txt_retry_icsi.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)

    txt_retry_dict = get_txt_retry_dict(dir_dict["SBCSAE text"], 5, 5, 32)
    with open("txt_retry_sbcsae.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)

    txt_retry_dict = get_txt_retry_dict(dir_dict["CHiME-5 text1"], 5, 5, 32)
    with open("txt_retry_chime5.json", 'w') as ami_out:
        json.dump(txt_retry_dict, ami_out, indent=4)
