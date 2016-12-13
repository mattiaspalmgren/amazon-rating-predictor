from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import json, nltk


def pre_process(documents):
    st = LancasterStemmer()
    stop = stopwords.words('english')
    stop.append('')

    doc_toks = []
    for document in documents[:10]:
        doc_toks.append(nltk.word_tokenize(document['reviewText']))

    # Remove stopwords and stem tokens
    tmp_doc_toks = []
    for doc_tok in doc_toks:
        tmp_doc_tok = []
        for tok in doc_tok:
            if tok not in stop:
                    tmp_doc_tok.append(st.stem(tok))

        tmp_doc_toks.append(tmp_doc_tok)

    return tmp_doc_toks

# Load data
reviews = []
for line in open('data/electronics-reviews-sub.json', 'r'):
    reviews.append(json.loads(line))

document_tokens = pre_process(reviews)
print(document_tokens[0])


