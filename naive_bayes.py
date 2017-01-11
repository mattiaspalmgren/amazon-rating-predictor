import numpy as np
from process import *
from features import *
from evaluate import *


# Load original data
# data = load_original_data()
# reviews = pre_process(data)

# Dump processed data
# with open('data/reviews_processed_without_stem.json', 'w') as outfile:
#     json.dump(reviews, outfile)

# Open processed data
reviews = load_processed_data('data/reviews_processed_without_stem.json')

# Subset data
random.seed(12345)
np.random.shuffle(reviews)
reviews = reviews[:10000]


# Build necessary helping structures
vocabulary = build_vocabulary(reviews)
# top_words = build_best_words(reviews)

# Different feature vectors

# reviews_features = [(counter_feature(text), c) for (text, c) in reviews]

# reviews_features = [(occurs_feature(text, vocabulary), c) for (text, c) in reviews]

bigram_features = [(bigram_feature(text), c) for (text, c) in reviews]

# top_word_features = [(top_word_feats(text, top_words), c) for (text, c) in reviews]

# reviews_features = combine_features(bigram_features, top_word_features)
reviews_features = bigram_features

# Divide into train and test set
train_reviews, test_reviews = train_test_set(reviews_features)

# Train classifier
print("Training classifier...")
classifier = nltk.NaiveBayesClassifier.train(train_reviews)

# Evaluate classifier
print("Evaluation metrics...")
evaluate_classifier(classifier, test_reviews)


