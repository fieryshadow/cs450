import random
import difflib
from sklearn import datasets

iris = datasets.load_iris()
answer_key = list(zip(iris.data, iris.target))
random.shuffle(answer_key)

split_point  = round(.7 * len(answer_key))
training_set = [entry[0] for entry in answer_key[:split_point]]
training_key = [entry[1] for entry in answer_key[:split_point]]
testing_set  = [entry[0] for entry in answer_key[split_point:]]
testing_key  = [entry[1] for entry in answer_key[split_point:]]

class HardCoded:

    def train(self, data, answers):
        pass

    def predict(self, data):
        return [0 for i in data]

learner = HardCoded()
learner.train(training_set, training_key)
prediction = learner.predict(testing_set)

accuracy = difflib.SequenceMatcher(None, prediction, testing_key).ratio()
percentage = int(round(100 * accuracy, 0))
print("The Hardcoded Classifier was {}% accurate.".format(percentage))
