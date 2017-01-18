from evaluate import *
from process import *
from dictionarytagger import *
from matplotlib import style
from sklearn import svm, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack
style.use("ggplot")

# Load original data
data = load_original_data()
# data = data[:50]

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
scores = np.array([score for (score, label) in scores])
scores = scores.reshape(-1, 1)

# TF-IDF representation
count_vect = CountVectorizer()
texts = [text for (text, c) in data if len(text) > 0]
X_counts = count_vect.fit_transform(texts)
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X_counts)

# Combine TF-IDF with sentiment score
X = hstack((scores, X))

y = [0 if label == 'neg' else 1 for (text, label) in data if len(text) > 0]

# Fit Linear SVM
classifier = svm.SVC(kernel='linear', C=1.0, random_state=42)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier, and predict on test samples
classifier.fit(X_train, y_train)
y_hat = classifier.predict(X_test)

# Evaluate classifier
print("Evaluation metrics...")
evaluate_svm(y_test, y_hat)


