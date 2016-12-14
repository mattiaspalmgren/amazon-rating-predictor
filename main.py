import json, nltk
from process import *

# Load data
reviews = []
for line in open('data/electronics-reviews-sub.json', 'r'):
    reviews.append(json.loads(line))

document_tokens = pre_process(reviews)
print(document_tokens[0])


