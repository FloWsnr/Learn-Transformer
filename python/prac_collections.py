from collections import Counter, defaultdict


def analyze_corpus(documents: list[str], top_k: int = 10, min_docs: int = 2) -> dict:
    """
    Returns {
        "top_words": [("word", count), ...],
        "by_letter": {"a": ["apple", "ant"], ...},
        "first_seen_order": ["the", "quick", ...]
    }
    """
    data = {"top_words": [], "by_letter": defaultdict(list), "first_seen_order": []}
    for doc in documents:
        counted_doc = Counter(doc)
        pass

    return data


docs = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy cat sleeps all day",
    "a quick red fox runs fast",
]
print(analyze_corpus(docs, top_k=5, min_docs=2))
