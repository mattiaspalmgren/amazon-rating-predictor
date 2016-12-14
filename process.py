import nltk, re, json
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


def read_documents(filename):
    documents = []
    for line in open (filename, 'r'):
        documents.append(json.loads(line))
    return documents


def pre_process(documents):
    document_tokens = [(nltk.word_tokenize(text), c) for (text, c) in documents]
    document_tokens = [clean_tokens(tokens, c) for (tokens, c) in document_tokens]

    return document_tokens


def clean_tokens(tokens, c):
    stop = stopwords.words('english')
    stop.append('')
    st = LancasterStemmer()

    tmp_tokens = []
    for token in tokens:
        token = st.stem(token)
        token = (re.sub(r'\W+', '', str(token)).lower())
        if token not in stop:
            tmp_tokens.append(token)

    return tmp_tokens, c


