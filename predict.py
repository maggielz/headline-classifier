from typing import List
from bayes import *
from whitelist import *
import probabilities
from preprocess import *
from csv_reader import *

class Predict:
    def __init__(self,training_set, filter_stop_words=True, stem_words=True, threshold_probability=0.02) -> None:
        self.word_to_category_probabilities, self.category_probabilities, self.default_probability = probabilities.get_probabilities(training_set,filter_stop_words,stem_words)
        self.filter_stop_words = filter_stop_words
        self.stem_words = stem_words
        self.whitelist_dictionary = probabilities.create_whitelist_dictionary(self.word_to_category_probabilities,threshold_probability)

    def bayes(self,headline: str):
        return bayes(preprocess(headline,self.filter_stop_words,self.stem_words),self.word_to_category_probabilities,self.category_probabilities,self.default_probability)

    def whitelist(self,headline: str):
        return whitelist(headline,self.whitelist_dictionary,self.filter_stop_words,self.stem_words)

    def __test(self,test_data: List[List[str]], labels: List, prediction_func):
        headlineIndex = labels.index('Headline')
        categoryIndex = labels.index('Category')

        TP = 0 # true positives
        FP = 0 # false positives
        TN = 0 # true negatives
        FN = 0 # false negatives

        for row in test_data:
            test_result = prediction_func(row[headlineIndex])
            true_category = Category[row[categoryIndex].upper().replace(' ','_')] # convert category names to enum

            for category in Category:
                if category == test_result and true_category == category:
                    TP += 1
                elif category == test_result and true_category != category:
                    FP += 1
                elif category != test_result and true_category != category:
                    TN += 1
                else:
                    FN += 1
        
        accuracy = (TP + TN)/(TP + FP + TN + FN)
        precision = TP/(TP + FP)

        return accuracy, precision

    def test_bayes(self,test_data: List[List[str]], labels: List):
        return self.__test(test_data,labels,self.bayes)

    def test_whitelist(self,test_data: List[List[str]], labels: List):
        return self.__test(test_data,labels,self.whitelist)