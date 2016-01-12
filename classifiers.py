import sys
import random
import difflib
from sklearn import datasets

class HardCoded:
    """A hard coded, same answer always algorithm"""
    def __init__(self):
        self.targets = []

    def train(self, data, answers):
        self.targets.extend(set(answers))

    def predict(self, data):
        return [self.targets[0] for i in data]

class kNN:
    """The k-Nearest Neighbors algorithm"""
    def __init__(self):
        self.targets = []

    def train(self, data, answers):
        self.targets.extend(set(answers))

    def predict(self, data):
        return [self.targets[0] for i in data] # placeholder


def load_from_file(filename):
    class DataHolder:
        def __init__(self, data, target, target_names):
            self.data = data
            self.target = target
            self.target_names = target_names

    return DataHolder([], [], []) # placeholder


dataset = learner = train_test_split = None
algorithm_matcher = { 'kNN': kNN, 'HC': HardCoded }
for arg in sys.argv[1:]:
    if arg.startswith('-A'):
        learner = algorithm_matcher[arg[2:]]()
    elif arg.startswith('-R'):
        train_test_split = float(arg[2:])
    elif arg.startswith('-D'):
        dataset_name = arg[2:]
        if dataset_name == 'iris':
            dataset = datasets.load_iris()
        elif dataset_name.startswith('file='):
            dataset = load_from_file(dataset_name[5:])

if dataset is None: dataset = datasets.load_iris()
if learner is None: learner = HardCoded()
if train_test_split is None: train_test_split = .7

answer_key = list(zip(dataset.data, dataset.target))
random.shuffle(answer_key)

split_point  = round(train_test_split * len(answer_key))
training_set = [entry[0] for entry in answer_key[:split_point]]
training_key = [entry[1] for entry in answer_key[:split_point]]
testing_set  = [entry[0] for entry in answer_key[split_point:]]
testing_key  = [entry[1] for entry in answer_key[split_point:]]

learner.train(training_set, training_key)
prediction = learner.predict(testing_set)

accuracy = difflib.SequenceMatcher(None, prediction, testing_key).ratio()
percentage = int(round(100 * accuracy))
print("The {} classifier was {}% accurate.".format(
    learner.__class__.__name__, percentage))
