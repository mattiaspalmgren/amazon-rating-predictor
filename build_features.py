from collections import Counter


def build_vocabulary(documents):
    vocabulary = set()
    for (text, c) in documents:
            vocabulary.update([token for token in text])

    return vocabulary


def occurs_feature(tokens, vocabulary):
    features = {}
    for word in vocabulary:
        features['contains({})'.format(word)] = (word in tokens)
    return features

def counter_feature(tokens):
    return dict(Counter(tokens))


def bigram_feature(tokens):
    return dict(Counter(tokens))
