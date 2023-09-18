import openai

openai.api_key = "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"


def get_fixed_sentences(input: str, model: str = "gpt-3.5-turbo"):
    prompt = """Add appropriate punctuation such as commas and periods to the following input, return only the fixed 
    sentences.
    An example will be:
    Hi my name is Jinho How are you doing -> Hi, my name is Jinho. How are you doing?
    Now here's the text to be fixed: """
    response = openai.ChatCompletion.create(
                  model=model,
                  messages=[{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"{prompt}{input}"}
                  ])
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    for _ in range(10):
        sentences = "im yirtse Hashem alright Ill see you later who you staying by over there"
        fixed_sentences = get_fixed_sentences(sentences, "gpt-4")
        print(fixed_sentences)