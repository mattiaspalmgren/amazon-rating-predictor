import numpy as np
from process import *
from features import *
from evaluate import *

# Load original data
# data = load_original_data()
# reviews = pre_process(data)

# Dump processed data
# with open('data/reviews_processed.json', 'w') as outfile:
#     json.dump(reviews, outfile)

# Open processed data
reviews = load_processed_data('data/reviews_processed.json')

# Subset data
# random.seed(12345)
# np.random.shuffle(reviews)
# reviews = reviews[:100]


# Build necessary helping structures
# top_words = build_best_words(reviews)

# Different feature vectors
print("Forming the features...")

# reviews_features = [(counter_feature(text), c) for (text, c) in reviews]
feat = FeatureBuilder(reviews)

# Use counter feature
# reviews_features = feat.form_counter_feature()

# Build vocabulary, to form occurs feature
# feat.build_vocabulary()
# reviews_features = feat.form_occurs_feature()

# Use Bigram features
bigram_feature = feat.form_bigram_feature()

# Build top words, in order to form top word features
feat.build_top_words()
top_word_feature = feat.form_top_words_feature()

reviews_features = feat.combine_features(top_word_feature, bigram_feature)

# Divide into train and test set
train_reviews, test_reviews = train_test_set(reviews_features)

# Train classifier
print("Training classifier...")
classifier = nltk.NaiveBayesClassifier.train(train_reviews)

# Evaluate classifier
print("Evaluation metrics...")
evaluate_classifier(classifier, test_reviews)


