import nltk, re, json, math, random, numpy as np
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


# Read documents from file
def read_documents(filename):
    documents = []
    for line in open (filename, 'r'):
        documents.append(json.loads(line))
    return documents


# Tokenize and stem a set of documents
def pre_process(documents):
    document_tokens = [(nltk.word_tokenize(text), c) for (text, c) in documents]
    document_tokens = [clean_tokens(tokens, c) for (tokens, c) in document_tokens]

    return document_tokens


# Remove stopwords and stem from a description
def clean_tokens(tokens, c):
    stop = stopwords.words('english')
    stop.append('')
    st = LancasterStemmer()

    tmp_tokens = []
    for token in tokens:
        token = st.stem(token)
        token = (re.sub(r'\W+', '', str(token)).lower())
        if token not in ['']:
            tmp_tokens.append(token)

    return tmp_tokens, c


# Split a text into tokens
def split(text):
    splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    sentences = splitter.tokenize(text)
    tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
    return tokenized_sentences


# Tag tokens with their tag depending on what kind of word it is
def pos_tag(sentences):
    pos = [nltk.pos_tag(sentence) for sentence in sentences]
    pos = [[(word.lower(), [postag]) for (word, postag) in sentence] for sentence in pos]
    return pos


# Load the original unprocessed data
def load_original_data(url='data/reviews_Electronics_15k.json'):
    data = read_documents(url)

    # Convert stars into pos/neg
    star_map = {1.0: "neg", 2.0: "neg", 3.0: "neu", 4.0: "pos", 5.0: "pos"}
    data = tuple([(document['reviewText'], star_map[document['overall']]) for document in data])

    pos_reviews = [(document, c) for (document, c) in data if c == 'pos']
    pos_reviews = pos_reviews[:1400]
    neg_reviews = [(document, c) for (document, c) in data if c == 'neg']
    neg_reviews = neg_reviews[:1400]

    reviews = pos_reviews + neg_reviews
    random.seed(12345)
    random.shuffle(reviews)

    return reviews


# Load already processed (cleaning and tokenized) data
def load_processed_data(url):
    with open(url, 'r') as infile:
        reviews = json.load(infile)

    return reviews


# Divide into train and test set from set of features
def train_test_set(features):
    count = len(features)
    train_count = math.floor(0.8 * count)
    random.seed(12345)
    all_lines = random.sample(range(0, count), count)
    train_lines = all_lines[0:train_count]
    test_lines = all_lines[train_count:]

    train_set = [features[i] for i in train_lines]
    test_set = [features[i] for i in test_lines]

    return train_set, test_set
