import re
from nltk.stem import WordNetLemmatizer
import nltk.data
from FileIO import FileIO

class TextProcessor(object):
    """description of class"""
    STOP_WORD_LOCATION = "resources/stopwords.txt"

    def __init__(self, *args, **kwargs):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return super().__init__(*args, **kwargs)

    def remove_non_characters(self, text):
        """Basic cleaning of texts."""

        # remove non-ascii, digit, special and punctuation characters and split into words
        words = re.split(r'\W+|\d', text)
        return words
        
    def to_lower(self, words):
        # All words to lower case
        words = [w.lower() for w in words]
        return words

    def remove_stopwords(self, words):
        # filter out stop words
        file_io = FileIO()
        stop_words = file_io.loadTextFile(self.STOP_WORD_LOCATION)
        words = [w for w in words if not w in stop_words]
        return words

    def lemmatize_words(self, words):
        # init lemmatizer
        lemmatizer = WordNetLemmatizer()
        #lemmatize variations
        lemmatized_words = [lemmatizer.lemmatize(word = word, pos = 'v') for word in words]
        return lemmatized_words

    def stringifyTokenArray(self, tokenArray):
        return " ".join(tokenArray)

    def convert_to_sentences(self, text):
        return self.tokenizer.tokenize(text)

    def preprocessor(self, text):
        processed_text = self.remove_non_characters(text)
        processed_text = self.to_lower(processed_text)
        processed_text = self.remove_stopwords(processed_text)
        processed_text = self.lemmatize_words(processed_text)
        return processed_text