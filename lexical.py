from process import *
import nltk
import yaml
from evaluate import *
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm, model_selection

# Load original data
data = load_original_data()
# data = data[:1500]

def split(text):
    splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    sentences = splitter.tokenize(text)
    tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
    return tokenized_sentences


def pos_tag(sentences):
    pos = [nltk.pos_tag(sentence) for sentence in sentences]
    pos = [[(word.lower(), [postag]) for (word, postag) in sentence] for sentence in pos]
    return pos


class DictionaryTagger(object):
    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(str(key)))

    def tag(self, postagged_sentences):
        # print(self.tag_sentence(postagged_sentences[1]))
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence):
        def sentiment_tag(w, t):
            if w in self.dictionary:
                t.append(self.dictionary[w.lower()][0])
                res = (w, t)
                return res
            else:
                res = (w, t)
                return res
        k = [sentiment_tag(word, tag) for (word, tag) in sentence]
        # print(k)
        return k


def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0


def sentiment_score(d):
    # Sum occurences of sentiment words and normalize on length of the document
    return sum([value_of(tag) for sentence in d for (lit, tags) in sentence for tag in tags])/len(d)

print("Splitting...")
splitted_sentences = [(split(text), c) for (text, c) in data if len(text) > 0]

print("Word tagging...")
pos_tags = [(pos_tag(splitted_sentence), c) for (splitted_sentence, c) in splitted_sentences]

print("Building dictionary...")
dict_tagger = DictionaryTagger(['data/positive-words.yml', 'data/negative-words.yml'])

print("Sentiment tagging...")
dict_tagged_docs = [(dict_tagger.tag(doc), c) for (doc, c) in pos_tags]

print("Assigning sentiment score...")
scores = [(sentiment_score(doc), c) for (doc, c) in dict_tagged_docs]

# Fit Linear SVM
X = np.array([score for (score, label) in scores])
X = X.reshape(-1, 1)
y = [0 if label == 'neg' else 1 for (score, label) in scores]

classifier = svm.SVC(kernel='linear', C=1.0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

classifier.fit(X_train, y_train)
y_hat = classifier.predict(X_test)

# Evaluate classifier
print("Evaluation metrics...")
compute_evaluation_metrics(y_test, y_hat)

# plt.scatter(X,y)
# plt.show()


