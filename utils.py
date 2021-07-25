import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')


def clean_text(text):
    """
        Clean the Raw Text.
    """

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = text.strip(' ')  # strip whitespaces
    text = text.lower()  # lowercase
    # text = stem_text(text)  # stemming
    text = remove_special_characters(text)  # remove punctuation and symbols
    text = remove_stopwords(text)  # remove stopwords
    # text.strip(' ') # strip whitespaces again?
    text = text.encode("ascii", "ignore")
    text = text.decode()

    return text


def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.
    """
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return props


# print(load_properties(filepath='ConfigFile.properties'))

# print(clean_text("Trump has largely laid the blame for economic headwinds on the Fed, openly criticizing its chairman, Jerome Powell, whom he appointed."))