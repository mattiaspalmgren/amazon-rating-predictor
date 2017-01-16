import yaml


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
        return k


# Compute value of single segment
def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0


# Sum sentiment score for whole document
def sentiment_score(d):
    # Sum occurences of sentiment words and normalize on length of the document
    return sum([value_of(tag) for sentence in d for (lit, tags) in sentence for tag in tags])/len(d)

