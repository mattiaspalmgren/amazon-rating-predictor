from collections import Counter
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import FreqDist, ConditionalFreqDist


class FeatureBuilder(object):
    def __init__(self, docs):
        self.documents = docs
        self.features = []
        self.vocabulary = []
        self.top_words = []

    @staticmethod
    def count(tokens):
        return dict(Counter(tokens))

    @staticmethod
    def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        return dict([(bigram, True) for bigram in bigrams])

    def occurs(self, tokens):
        features = {}
        for word in self.vocabulary:
            features['contains({})'.format(word)] = (word in tokens)
        return features

    def filter_out_top_words(self, tokens):
        return dict([(word, True) for word in tokens if word in self.top_words])

    def build_vocabulary(self):
        v = set()
        for (text, c) in self.documents:
            v.update([token for token in text])

        self.vocabulary = v

    def build_top_words(self):
        pos_reviews = [(review, c) for (review, c) in self.documents if c == 'pos']
        neg_reviews = [(review, c) for (review, c) in self.documents if c == 'neg']

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

        best = sorted(word_scores.items(), reverse=True, key=lambda x: x[1])[:1000]
        self.top_words = set([w for w, s in best])

    def form_counter_feature(self):
        return [(self.count(text), c) for (text, c) in self.documents]

    def form_occurs_feature(self):
        return [(self.occurs(text), c) for (text, c) in self.documents]

    def form_bigram_feature(self):
        return [(self.bigram(text), c) for (text, c) in self.documents]

    def form_top_words_feature(self):
        return [(self.filter_out_top_words(text), c) for (text, c) in self.documents]

    def combine_features(self, f1, f2):
        combined_features = f1
        for (i, idx) in enumerate(range(0, len(f1))):
            combined_features[idx][0].update(f2[idx][0])
        return combined_features
