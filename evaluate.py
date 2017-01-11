from nltk import precision, recall
from sklearn.metrics import zero_one_loss
import collections, random, math


def evaluate_classifier(classifier, test_reviews):
    # For calculating precision/recall
    ref_set = collections.defaultdict(set)
    test_set = collections.defaultdict(set)
    ref_set_arr = []
    test_set_arr = []

    # Create gold standard and predicted labels
    for i, (feat, label) in enumerate(test_reviews):
        ref_set[label].add(i)
        observed = classifier.classify(feat)
        test_set[observed].add(i)
        ref_set_arr.append(label)
        test_set_arr.append(observed)

    print('neg precision:', precision(ref_set['neg'], test_set['neg']))
    print('pos precision:', precision(ref_set['pos'], test_set['pos']))
    print('neg recall:', recall(ref_set['neg'], test_set['neg']))
    print('pos recall:', recall(ref_set['pos'], test_set['pos']))
    print('misclassification rate', zero_one_loss(ref_set_arr, test_set_arr))
    # print('most informative features', classifier.show_most_informative_features(10))
    return(0)


