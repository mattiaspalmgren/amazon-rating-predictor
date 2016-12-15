from collections import Counter
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import FreqDist, ConditionalFreqDist


def build_vocabulary(documents):
    vocabulary = set()
    for (text, c) in documents:
            vocabulary.update([token for token in text])

    return vocabulary


def build_best_words(reviews):
    pos_reviews = [(review, c) for (review, c) in reviews if c == 'pos']
    neg_reviews = [(review, c) for (review, c) in reviews if c == 'neg']

    pos_words = [token for (review, c) in pos_reviews for token in review]
    neg_words = [token for (review, c) in neg_reviews for token in review]

    fd_all = FreqDist(pos_words + neg_words)
    pos_class_words = [('pos', word) for word in pos_words]
    neg_class_words = [('neg', word) for word in neg_words]
    cfd_pos = ConditionalFreqDist(pos_class_words)
    cfd_neg = ConditionalFreqDist(neg_class_words)

    pos_word_count = len(pos_words)
    neg_word_count = len(neg_words)
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}

    for (word, freq) in fd_all.items():
        pos_score = BigramAssocMeasures.chi_sq(cfd_pos['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cfd_neg['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    best = sorted(word_scores.items(), reverse=True, key=lambda x: x[1])[:10000]
    return set([w for w, s in best])


def occurs_feature(tokens, vocabulary):
    features = {}
    for word in vocabulary:
        features['contains({})'.format(word)] = (word in tokens)
    return features


def counter_feature(tokens):
    return dict(Counter(tokens))


def bigram_feature(words, score_fn=BigramAssocMeasures.chi_sq, n=200):

    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    return d

def best_word_feats(words, best_words):
    return dict([(word, True) for word in words if word in best_words])

