from categories import *

def bayes(input_list, word_probability_dict, category_probability_dict, defaultProbability):
    """
    input_list: [word]
        List of tokenized words in the headline

    word_probability_dict: { category : { word : P(word | category) }}
        P(word|category) contained in a dictionary

    category_probability_dict: {category: P(category)}
        P(category) contained in a dictionary

    defaultProbability: float
        The probability of a word not in the training set contained in the data,
        equal to 1/(number of words in training set + number of unique words in training set)
    
    """

    probability_calcs = []

    for category in Category:
        probability_calculation = category_probability_dict[category]
        for word in input_list:
            word_probability = word_probability_dict.get(word, defaultProbability)
            if isinstance(word_probability,dict):
                probability_calculation *= word_probability[category]
            else:
                probability_calculation *= word_probability

        probability_calcs.append(probability_calculation)

    return Category(probability_calcs.index(max(probability_calcs)))
