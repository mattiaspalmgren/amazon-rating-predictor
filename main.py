from nltk import precision, recall
import collections, random, math
from process import *
from build_features import *

# Load data
data = read_documents('data/electronics-reviews-sub.json')

# Convert stars into pos/neg
star_map = {1.0: "neg", 2.0: "neg", 3.0: "pos", 4.0: "pos", 5.0: "pos"}
data = tuple([(document['reviewText'], star_map[document['overall']]) for document in data[:1000]])
reviews = pre_process(data)

# print(reviews[0])

vocalbulary = build_vocabulary(reviews)

reviews_features = [(counter_feature(text), c) for (text, c) in reviews]
print(reviews_features)

# reviews_features = [(occurs_feature(text, vocalbulary), c) for (text, c) in reviews]
# print(reviews_features)

random.seed(12345)
COUNT = len(reviews_features)
TRAIN_COUNT = math.floor(0.8 * COUNT)
TEST_COUNT = math.floor(0.2 * COUNT)

all_lines = random.sample(range(0, COUNT), COUNT)
train_lines = all_lines[0:TRAIN_COUNT]
test_lines = all_lines[TRAIN_COUNT:]

train_reviews = [reviews_features[i] for i in train_lines]
test_reviews = [reviews_features[i] for i in test_lines]

# print(len(train_reviews))
# print(len(test_reviews))

classifier = nltk.NaiveBayesClassifier.train(train_reviews)

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
