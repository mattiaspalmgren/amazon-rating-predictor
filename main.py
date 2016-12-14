from collections import Counter
from process import *
import linecache, random, math

# Load data
data = read_documents('data/electronics-reviews-sub.json')

# Convert stars into pos/neg
star_map = {1.0: "neg", 2.0: "neg", 3.0: "pos", 4.0: "pos", 5.0: "pos"}
data = tuple([(document['reviewText'], star_map[document['overall']]) for document in data[:1000]])

reviews = pre_process(data)
reviews = [(dict(Counter(text)), c) for (text, c) in reviews]

# print(reviews)

COUNT = len(reviews)
TRAIN_COUNT = math.floor(0.8 * COUNT)
TEST_COUNT = math.floor(0.2 * COUNT)

all_lines = random.sample(range(0, COUNT), COUNT)
train_lines = all_lines[0:TRAIN_COUNT]
test_lines = all_lines[TRAIN_COUNT:]

train_reviews = [reviews[i] for i in train_lines]
test_reviews = [reviews[i] for i in test_lines]

# print(train_reviews[0])
# print(test_reviews[0])

classifier = nltk.NaiveBayesClassifier.train(train_reviews)

guess = []
correct = 0
incorrect = 0
for (feature, c) in test_reviews:
    guess = classifier.classify(feature)
    if guess == c:
        correct += 1
    else:
        incorrect += 1

print(correct, incorrect)
