import os
import re
from multiprocessing import Pool

from numba import njit

import whisper

from pydub import AudioSegment
from pydub.silence import split_on_silence

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


def transcribe(audio_dir: str, text_output_dir: str):
    """
    Transcribe the audios with whisper.
    :param audio_dir: Parent directory of all audio files.
    :param text_output_dir: Parent directory of all text output files.
    :return: None.
    """
    model = whisper.load_model("large")
    for root, dirnames, filenames in os.walk(audio_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == ".wav":
                with open(os.path.join(text_output_dir, basename + ".txt"), 'w', encoding="utf-8",
                          errors="ignore") as output_file:
                    result = model.transcribe(os.path.join(root, filename))
                    output_file.write(result["text"])


@njit()
def is_repeating(input_str: str, min_length: int, min_repeat: int) -> bool:
    """
    Check if a string contains substring that is repeating itself.
    :param input_str: String need to check.
    :param min_length: Minimum length of the repeating substring.
    :param min_repeat: Minimum repeating times of the repeating substring.
    :return: boolean.
    """
    max_length = len(input_str) // min_length
    for length in range(max_length, min_length - 1, -1):
        for start in range(len(input_str) - length * min_repeat + 1):
            substring = input_str[start:start + length]
            pattern = substring * min_repeat
            if pattern in input_str:
                return True
    return False


def is_repeating_file(txt_filename: str, min_length: int, min_repeat: int) -> bool:
    with open(txt_filename, 'r', encoding="utf-8", errors="ignore") as txt:
        return is_repeating(txt.read(), min_length, min_repeat)


def is_repeating_file_parallel(args: tuple) -> tuple | None:
    txt_root, txt_filename, min_length, min_repeat = args
    if is_repeating_file(os.path.join(txt_root, txt_filename), min_length, min_repeat):
        return os.path.splitext(txt_filename)[0], os.path.join(txt_root, txt_filename)
    return None


def retranscribe(audio_dir: str, txt_retry_dict: dict):
    model = whisper.load_model("large")
    for wav_root, wav_dirnames, wav_filenames in os.walk(audio_dir):
        for wav_filename in wav_filenames:
            wav_basename = os.path.splitext(wav_filename)[0]
            if wav_basename in txt_retry_dict:  # check if key exists
                with open(txt_retry_dict[wav_basename], 'w', encoding="utf-8", errors="ignore") as new_txt:
                    retry_result = model.transcribe(os.path.join(wav_root, wav_filename))
                    new_txt.write(retry_result["text"])
                print(f"Retry: {wav_basename}")


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


def retry_error_file(audio_dir: str, text_output_dir: str, min_length: int, min_repeat: int):
    """
    Re-transcribe audio files with repeating in the text transcription.
    :param audio_dir: Parent directory of all audio files.
    :param text_output_dir: Parent directory of all text output files.
    :param min_length: Minimum length that a sequence of text can be recognized to be repeated.
    :param min_repeat: Minimum times of repetition for a sequence of text.
    :return: None
    """
    txt_retry_dict = {}  # key: basename, value: path(root + filename)
    for txt_root, txt_dirnames, txt_filenames in os.walk(text_output_dir):
        for txt_filename in txt_filenames:
            if is_repeating_file(os.path.join(txt_root, txt_filename), min_length, min_repeat):
                txt_retry_dict[os.path.splitext(txt_filename)[0]] = os.path.join(txt_root, txt_filename)

    retranscribe(audio_dir, txt_retry_dict)


def retry_error_file_parallel(audio_dir: str, text_output_dir: str, min_length: int, min_repeat: int, n_process: int = 4):
    """
    This is a parallel version of retry_error_file using processpool from multiprocessing.
    This only parallels the part of detecting repeating files on the file level, which uses CPU only.
    :param audio_dir: Parent directory of all audio files.
    :param text_output_dir: Parent directory of all text output files.
    :param min_length: Minimum length that a sequence of text can be recognized to be repeated.
    :param min_repeat: Minimum times of repetition for a sequence of text.
    :param n_process: Number of precesses it used in detecting repetition for multiple files.
    :return: None
    """
    txt_retry_dict = get_txt_retry_dict(text_output_dir, min_length, min_repeat, n_process)
    retranscribe(audio_dir, txt_retry_dict)


def remove_silence_from_audio(file_path, output_file_path, silence_thresh=-50.0, min_silence_len=1000, keep_silence=100):
    """
    Remove silence from an audio file.
    Instead of modifying the original file, this function produce an additional audio file with silence removal.
    Remember to remove it after using.
    :param file_path: path to the audio file.
    :param output_file_path: path to save the processed audio.
    :param silence_thresh: in dB. Anything quieter than this will be considered silence. (Default is -50.0 dB).
    :param min_silence_len: minimum length of silence in milliseconds to be considered silence. (Default is 1000ms = 1 second).
    """
    audio = AudioSegment.from_file(file_path, format="wav")
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence)
    audio_without_silence = sum(chunks, AudioSegment.empty())
    audio_without_silence.export(output_file_path, format="wav")


def retranscribe_with_silence_removal(audio_dir: str, txt_retry_dict: dict, silence_thresh=-50.0, min_silence_len: int=1000, keep_silence: int=100):
    model = whisper.load_model("large")
    for wav_root, wav_dirnames, wav_filenames in os.walk(audio_dir):
        for wav_filename in wav_filenames:
            wav_basename, wav_ext = os.path.splitext(wav_filename)
            if wav_basename in txt_retry_dict:  # check if key exists
                with open(txt_retry_dict[wav_basename], 'w', encoding="utf-8", errors="ignore") as new_txt:
                    wav_filepath = os.path.join(wav_root, wav_filename)
                    wav_filepath_no_silence = os.path.join(wav_root, wav_basename + "_no_silence" + wav_ext)
                    remove_silence_from_audio(wav_filepath, wav_filepath_no_silence, silence_thresh, min_silence_len, keep_silence)
                    retry_result = model.transcribe(wav_filepath_no_silence)
                    new_txt.write(retry_result["text"])
                    os.remove(wav_filepath_no_silence)  # remove no silence audio
                print(f"Retry: {wav_basename}")


def retry_with_silence_removal(audio_dir: str, text_output_dir: str, min_length: int, min_repeat: int,
                               n_process: int = 4, silence_thresh=-50.0, min_silence_len=1000, keep_silence=100):
    txt_retry_dict = get_txt_retry_dict(text_output_dir, min_length, min_repeat, n_process)
    retranscribe_with_silence_removal(audio_dir, txt_retry_dict, silence_thresh, min_silence_len, keep_silence)


if __name__ == "__main__":
    # retry_with_silence_removal(dir_dict["AMI audio"], dir_dict["AMI text"], 5, 5, 4)
    # retry_with_silence_removal(dir_dict["CallFriend audio"], dir_dict["CallFriend text"], 5, 5, 4)
    # retry_with_silence_removal(dir_dict["CallHome English audio"], dir_dict["CallHome English text"], 5, 5, 4)
    retry_with_silence_removal(dir_dict["CHiME-5 audio1"], dir_dict["CHiME-5 text1"], 5, 5, 4, silence_thresh=-45.0)  # running
    # retry_with_silence_removal(dir_dict["CHiME-5 audio2"], dir_dict["CHiME-5 text2"], 5, 5, 4)
    # retry_with_silence_removal(dir_dict["DailyTalk audio"], dir_dict["DailyTalk text"], 5, 5, 4)
    # retry_with_silence_removal(dir_dict["ICSI audio"], dir_dict["ICSI text"], 5, 5, 4)
    # retry_with_silence_removal(dir_dict["SBCSAE audio"], dir_dict["SBCSAE text"], 5, 5, 4)
