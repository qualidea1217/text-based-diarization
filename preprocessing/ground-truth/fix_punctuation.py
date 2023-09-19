import json
import re

import openai


def fix_utterance_gpt_chat(input: str, api_key: str, model: str = "gpt-4"):
    openai.api_key = api_key
    prompt = """Add appropriate punctuation such as commas and periods to the following input text, return only the 
        fixed sentences. If the input text is too short or already good to go, just return "<NO NEED TO CHANGE>".
        An example will be:
        Hi my name is Jinho How are you doing -> Hi, my name is Jinho. How are you doing?
        Now here's the text to be fixed: """
    response = openai.ChatCompletion.create(
                  model=model,
                  messages=[{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"{prompt}{input}"}
                  ])
    return response["choices"][0]["message"]["content"]


def fix_utterance_rule(utterance: str):
    return utterance


def fix_punctuation(input_json: str, output_json: str, api_key: str, model: str = "gpt-4"):
    with open(input_json, 'r') as json_in:
        input_dict = json.load(json_in)
    text_list_raw = input_dict["text_list"]
    for conversation in text_list_raw:
        conversation_fix = []
        for utterance in conversation:
            utterance_fix = fix_utterance_gpt_chat(utterance, api_key, model)
            if "<NO NEED TO CHANGE>" in utterance_fix:
                conversation_fix.append(utterance)
                print()
                print(utterance_fix)
                print(utterance)
                print(len(utterance.split()))
                print()
            else:
                conversation_fix.append(utterance_fix)
                print(utterance_fix)
        break


if __name__ == "__main__":
    fix_punctuation("dataset7_gt_test.json", "", "")