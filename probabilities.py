import csv
from whitelist import whitelist
from categories import *
from preprocess import preprocess
import string

"""
Calculates probabilities of word given category using data in the provided training set.
"""
def get_probabilities (data_array, filter_stop_words, stem_words):
    p_word_given_category = {}
    p_category = {}
    total_count = 0
    words_per_category = {}
    total_words = 0

    # Initialize values to 0
    for category in Category:
        p_category[category] = 0
        words_per_category[category] = 0

    for row in data_array:
        total_count += 1
        category_string = row[1]

        # get category enum value
        category = Category[category_string.upper().replace(' ', '_')]

        p_category[category] += 1
                
        headline = row[0]                
        words = preprocess (headline, filter_stop_words, stem_words)
        # For each word, increment the total number of words present in headlines in the category
        # and increment the number of appearances of the particular word in the category.
        # If there is not a dictionary already set up for that word, we need to create one.
        total_words += len(words)
        for word in words:
            words_per_category [category] += 1
            if word in p_word_given_category:
                p_word_given_category[word][category] += 1
            else:
                dict_category_to_p = {}
                for dict_category in Category:
                    dict_category_to_p[dict_category] = 0
                p_word_given_category[word] = dict_category_to_p
                p_word_given_category[word][category] += 1

    # Divide all the counts by totals to get probabilities

    for item in p_category:
        p_category [item] /= total_count
    
    for word_dict in p_word_given_category:
        for category_dict in p_word_given_category[word_dict]:
            # Add one to the count of every word in every category so that we don't end up with 0% chances
            p_word_given_category[word_dict][category_dict] += 1
            if words_per_category[category_dict] == 0:
                p_word_given_category[word_dict][category_dict] = 0
            else:
                p_word_given_category[word_dict][category_dict] /= words_per_category[category_dict]

    return p_word_given_category, p_category, 1 / total_words

def create_whitelist_dictionary(word_dict, threshold_probability=0.02):
    whitelist_dict = {}

    for word in word_dict:
        if word in string.punctuation:
            continue
        max_prob = 0
        for category in word_dict[word]:
            if word_dict[word][category] >= threshold_probability:
                if word_dict[word][category] > max_prob:
                    max_prob = word_dict[word][category]
                    whitelist_dict[word] = category

    return whitelist_dict
