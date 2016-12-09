import json
from pprint import pprint

reviews = []
for line in open('data/electronics-reviews-sub.json', 'r'):
    reviews.append(json.loads(line))

print(reviews[0]['summary'])