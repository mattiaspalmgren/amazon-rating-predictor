from nltk import precision, recall
import collections, random, math
import numpy as np

from process import *
from features import *


# Load original data
# data = load_original_data()
# reviews = pre_process(data)

# Dump processed data
# with open('data/reviews_processed_without_stem.json', 'w') as outfile:
#     json.dump(reviews, outfile)

# Open
reviews = load_processed_data('data/reviews_processed_without_stem.json')
# random.seed(12345)
# np.random.shuffle(reviews)
# reviews = reviews[:1000]


vocabulary = build_vocabulary(reviews)
best_words = build_best_words(reviews)

# reviews_features = [(counter_feature(text), c) for (text, c) in reviews]

# reviews_features = [(occurs_feature(text, vocabulary), c) for (text, c) in reviews]

reviews_features = [(bigram_feature(text), c) for (text, c) in reviews]

# reviews_features = [(best_word_feats(text, best_words), c) for (text, c) in reviews]

# print(reviews_features)

# Divide into train and test set
train_reviews, test_reviews = train_test_set(reviews_features)

# Train classifier
classifier = nltk.NaiveBayesClassifier.train(train_reviews)

# Most informative feature
print(classifier.show_most_informative_features(32))

# For calculating precision/recall
ref_set = collections.defaultdict(set)
test_set = collections.defaultdict(set)

for i, (feat, label) in enumerate(test_reviews):
    ref_set[label].add(i)
    observed = classifier.classify(feat)
    test_set[observed].add(i)

print('neg precision:', precision(ref_set['neg'], test_set['neg']))
print('pos precision:', precision(ref_set['pos'], test_set['pos']))
print('neg recall:', recall(ref_set['neg'], test_set['neg']))
print('pos recall:', recall(ref_set['pos'], test_set['pos']))
