"""basic implementation of Byte-Pair-Encoding"""

from collections import Counter, defaultdict


def bpe(text: str, vocab_size: int) -> dict:
    text_bytes = list(text.encode("utf-8"))
    vocab = defaultdict(int)
    for i in range(256):
        vocab[i] = i

    while len(vocab) < vocab_size:
        pairs = zip(text_bytes, text_bytes[1:])  # [(t0,t1), (t1,t2), ...]
        counted = Counter(pairs)
        if not counted:
            break
        most_freq, _ = counted.most_common(1)[0]

        new_id = len(vocab)
        vocab[new_id] = most_freq

        # replace old tuple of ids with new id
        new_text_bytes = []
        pos = 0
        while pos < len(text_bytes):
            t0 = text_bytes[pos]
            if pos < len(text_bytes) - 1:
                t1 = text_bytes[pos + 1]
                if (t0, t1) == most_freq:
                    new_text_bytes.append(new_id)
                    pos += 2
                else:
                    new_text_bytes.append(t0)
                    pos += 1

            else:
                new_text_bytes.append(t0)
                pos += 1

        text_bytes = new_text_bytes

    return vocab


if __name__ == "__main__":
    text = "Think out loud constantly. They're evaluating your reasoning as much as your code."
    bpe(text, 300)
