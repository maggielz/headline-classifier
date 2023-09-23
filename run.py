from csv_reader import read_data
from preprocess import preprocess
from categories import Category
from predict import Predict

training_set, labels = read_data("training.csv")
threshold = 0.0
predictor = Predict(training_set,True,True, threshold)

data, labels = read_data('Polygon Article Categorization Dataset.csv')

print(predictor.bayes("Pokémon Go guide: August 2021 research tasks and breakthrough reward"))

print("Bayes accuracy + precision:",predictor.test_bayes(data,labels))

print("Whitelist accuracy + precision:", predictor.test_whitelist(data,labels))