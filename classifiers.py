import sys
import random
import numpy as np
from sklearn import datasets, metrics

class HardCoded:
    """A hard coded, same answer always algorithm"""
    def __init__(self):
        self.targets = []

    def train(self, data, answers):
        self.targets.extend(set(answers))

    def predict(self, data):
        return [self.targets[0] for i in data]

class KNearestNeighbors:
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

def cross_val_score(classifier, data, target, folds):
    k = max(0, min(folds, len(data)))
    step = len(data) // k
    results = np.zeros(k)
    for i, pos in zip(range(0, len(data), step), range(k)):
        train_set = data[:i] + data[i+step:]
        test_set = data[i:i+step]
        train_key = target[:i] + target[i+step:]
        test_key = target[i:i+step]

        learner = classifier()
        learner.train(train_set, train_key)
        prediction = learner.predict(test_set)
        results[pos] = metrics.accuracy_score(test_key, prediction)
    return results

def main():
    dataset = classifier = train_test_split = k_folds = None
    algorithm_matcher = { 'kNN': KNearestNeighbors, 'HC': HardCoded }
    for arg in sys.argv[1:]:
        if arg.startswith('-A'):
            classifier = algorithm_matcher[arg[2:]]
        elif arg.startswith('-S'):
            train_test_split = float(arg[2:])
        elif arg.startswith('-D'):
            dataset_name = arg[2:]
            if dataset_name == 'iris':
                dataset = datasets.load_iris()
            elif dataset_name.startswith('file='):
                dataset = load_from_file(dataset_name[5:])
        elif arg.startswith('-C'):
            k_folds = int(arg[2:])

    if dataset is None: dataset = datasets.load_iris()
    if classifier is None: classifier = HardCoded
    collected_data = list(zip(dataset.data, dataset.target))
    random.shuffle(collected_data)
    data, target = zip(*collected_data)

    if k_folds and k_folds > 1:
        accuracy = cross_val_score(classifier, data, target, k_folds).mean()
    else:
        if train_test_split is None: train_test_split = .7
        learner = classifier()

        split_point  = round(train_test_split * len(data))
        train_set =   data[:split_point]
        train_key = target[:split_point]
        test_set  =   data[split_point:]
        test_key  = target[split_point:]

        learner.train(train_set, train_key)
        prediction = learner.predict(test_set)
        accuracy = metrics.accuracy_score(test_key, prediction)

    percentage = int(round(100 * accuracy))
    print("The {} classifier was {}% accurate.".format(
        classifier.__name__, percentage))

if __name__ == '__main__':
    main()
