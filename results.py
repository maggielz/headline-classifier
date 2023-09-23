from csv_reader import *
from cross_validation import *

# preprocess data into folds
data, labels = read_data('dataset_v2.csv')
folds = preprocess_data(data, 10)

output = ""

# perform cross validation - whitelist - filter
train_acc, valid_acc, train_prec, valid_prec = cross_validation(folds, labels, [0], True, True)
print("====================", end='')
print("whitelist, filter")
print("train_acc, valid_acc, train_prec, valid_prec: ", train_acc, valid_acc, train_prec, valid_prec)
output += str(valid_acc[0]) + " & " + str(valid_prec[0]) + " \\\\\n"

# perform cross validation - whitelist - no filter
train_acc, valid_acc, train_prec, valid_prec = cross_validation(folds, labels, [0], False, True)
print("====================", end='')
print("whitelist, no filter")
print("train_acc, valid_acc, train_prec, valid_prec: ", train_acc, valid_acc, train_prec, valid_prec)
output += str(valid_acc[0]) + " & " + str(valid_prec[0]) + " \\\\\n"

# perform cross validation - bayes - filter
train_acc, valid_acc, train_prec, valid_prec = cross_validation(folds, labels, [0], True, False)
print("====================", end='')
print("bayes, filter")
print("train_acc, valid_acc, train_prec, valid_prec: ", train_acc, valid_acc, train_prec, valid_prec)
output += str(valid_acc[0]) + " & " + str(valid_prec[0]) + " \\\\\n"

# perform cross validation - bayes - no filter
train_acc, valid_acc, train_prec, valid_prec = cross_validation(folds, labels, [0], False, False)
print("====================", end='')
print("bayes, no filter")
print("train_acc, valid_acc, train_prec, valid_prec: ", train_acc, valid_acc, train_prec, valid_prec)
output += str(valid_acc[0]) + " & " + str(valid_prec[0]) + " \\\\\n"

print("")
print(output)