import nltk, re
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


def pre_process(documents):
    st = LancasterStemmer()
    stop = stopwords.words('english')
    stop.append('')

    document_tokens = []
    for document in documents[:10]:
        document_tokens.append(nltk.word_tokenize(document['reviewText']))

    # Remove stopwords and stem tokens
    tmp_document_tokens = []
    for document_token in document_tokens:
        tmp_document_token = []
        for token in document_token:
            # Remove non alpha-numeric characters and lowercase words
            token = re.sub(r'\W+', '', str(token)).lower()
            if token not in stop:
                # Stem token
                token = st.stem(token)
                tmp_document_token.append(token)

    tmp_document_tokens.append(tmp_document_token)

    return tmp_document_tokens
