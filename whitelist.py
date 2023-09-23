from categories import *
from random import randrange
from preprocess import *

# whitelist: headline, whitelist_dictionary -> category
#   string, (string, string) -> category enum
def whitelist(headline, whitelist_dictionary, filter_stop_words=True, stem_words=True):
    words = preprocess(headline, filter_stop_words, stem_words)

    for word in words:
        if word in whitelist_dictionary:
            return whitelist_dictionary[word]
    
    rand = randrange(0, NUM_CATEGORIES)
    # print("random: ")
    return Category(rand)