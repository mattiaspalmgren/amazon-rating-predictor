from nltk import precision, recall
from sklearn.metrics import zero_one_loss, precision_score, recall_score
import collections


def evaluate_classifier(classifier, test_reviews):
    # For calculating precision/recall
    ref_set = collections.defaultdict(set)
    test_set = collections.defaultdict(set)
    ref_set_arr = []
    test_set_arr = []

    # Create gold standard and predicted labels
    for i, (feat, label) in enumerate(test_reviews):
        # Predict
        observed = classifier.classify(feat)

        ref_set[label].add(i)
        test_set[observed].add(i)

        label = 0 if label == "neg" else 1
        observed = 0 if observed == "neg" else 1
        ref_set_arr.append(label)
        test_set_arr.append(observed)

    print('precision:', precision_score(ref_set_arr, test_set_arr))
    print('recall:', recall_score(ref_set_arr, test_set_arr))
    # print('neg precision:', precision(ref_set['neg'], test_set['neg']))
    # print('pos precision:', precision(ref_set['pos'], test_set['pos']))
    # print('neg recall:', recall(ref_set['neg'], test_set['neg']))
    # print('pos recall:', recall(ref_set['pos'], test_set['pos']))
    print('misclassification rate', zero_one_loss(ref_set_arr, test_set_arr))
    # print('most informative features', classifier.show_most_informative_features(10))


def compute_evaluation_metrics(ref_set, test_set):
    print('precision:', precision_score(ref_set, test_set))
    print('recall:', recall_score(ref_set, test_set))
    print('misclassification rate', zero_one_loss(ref_set, test_set))
