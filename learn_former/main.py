import tiktoken


def test_tokenizer(scentence):
    enc = tiktoken.get_encoding("cl100k_base")
    tokenized = enc.encode(scentence)
    print(tokenized)


if __name__ == "__main__":
    scentence = "The quick brown fox jumps over the lazy dog. This is a lengthier sentence that can be used for testing purposes."
    test_tokenizer(scentence)
