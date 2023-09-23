import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

"""
Tokenizes and otherwise preprocesses phrase
phrase: the string to be processed
filter_stop_words: if true, stop words (as defined by nltk) will be removed
stem_words: if true, words will be stemmed (endings removed, etc). 
"""
def preprocess (phrase: str, filter_stop_words: bool, stem_words: bool):
    # Split phrase into indiviudal tokens
    # Contractions may get split weirdly and sometimes inconsistently?
    tokens = nltk.word_tokenize (phrase)
    
    if filter_stop_words:
        # Removes anything from the nltk stop words list
        stop_words = list (stopwords.words ("english"))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        tokens = filtered_tokens

    if stem_words:
        # Stemming words leads to weird looking word stems like "commun" for "community"
        #   but it's consistent between words so the algorithm shouldn't care
        ps = PorterStemmer ()
        stemmed_tokens = [ps.stem (word) for word in tokens]
        tokens = stemmed_tokens
    else:
        # Stemming words converts to lower case; if we aren't stemming, we need to do it manually
        tokens = [word.lower () for word in tokens]

    return tokens
