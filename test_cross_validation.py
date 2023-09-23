from csv_reader import read_data
from preprocess import preprocess
from categories import Category
from predict import Predict
from cross_validation import *
from probabilities import *

# preprocess data into folds
data, labels = read_data('Polygon Article Categorization Dataset.csv')
folds = preprocess_data(data, 5)

# create paramater list from 0 to 0.095, going up by 0.005
value_list = []
for i in range(20):
    value_list.append(i / 200)
print(value_list)

# perform cross validation
train_acc, valid_acc, train_prec, valid_prec = cross_validation(folds, labels, value_list)

print(train_acc)
print("")
print(valid_acc)
print("")
print(train_prec)
print("")
print(valid_prec)

maximum = 0
idx = -1
for i in range(len(value_list)):
    if valid_acc[i] > maximum:
        idx = i
        maximum = valid_acc[i]
print("threshold_value, max_accuracy: ", value_list[idx], maximum)
