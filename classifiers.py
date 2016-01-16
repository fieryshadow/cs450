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

def make_k_nearest(k):
    def KNearestNeighbors():
        return _KNearestNeighbors(k)
    return KNearestNeighbors

class _KNearestNeighbors:
    """The k-Nearest Neighbors algorithm"""
    def __init__(self, k):
        self.k = k
        self.targets = []

    def train(self, data, answers):
        self.targets = list(zip(data, answers))

    def predict(self, data):
        k = self.k
        prediction = []
        t_data = self.targets
        for item in data:
            k_nearest = []
            for t_item, t_answer in t_data:
                rank = ((item - t_item)**2).sum()
                if len(k_nearest) < k or rank < k_nearest[-1][0]:
                    k_nearest.append((rank, t_item, t_answer))
                    k_nearest = sorted(k_nearest, key=lambda x:x[0])
                    if len(k_nearest) > k:
                        k_nearest.pop()
            possibles = [e[2] for e in k_nearest]
            prediction.append(max(possibles, key=possibles.count))
        return prediction


def load_from_file(filename):
    class DataHolder:
        def __init__(self, data, target, target_names):
            self.data = data
            self.target = target
            self.target_names = target_names

    return DataHolder([], [], []) # placeholder

def parse_args():
    collected_data = classifier = data_split = k_folds = None
    algorithm_matcher = { 'kNN': make_k_nearest, 'HC': HardCoded }
    for arg in sys.argv[1:]:
        if arg.startswith('-A='):
            algorithm = arg[3:]
            if ':' in algorithm:
                name, value = algorithm.split(':')
                classifier = algorithm_matcher[name](int(value or 3))
            else:
                classifier = algorithm_matcher[algorithm]
        elif arg.startswith('-S='):
            data_split = float(arg[3:] or .7)
        elif arg.startswith('-D='):
            dataset_name = arg[3:]
            if dataset_name == 'iris':
                collected_data = datasets.load_iris()
            elif dataset_name.startswith('file:'):
                collected_data = load_from_file(dataset_name[5:])
        elif arg.startswith('-C='):
            k_folds = int(arg[3:] or 0)

    if collected_data is None: collected_data = datasets.load_iris()
    if classifier is None: classifier = HardCoded
    if data_split is None: data_split = .7
    return collected_data, classifier, data_split, k_folds

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

def train_test_score(classifier, data, target, split_ratio):
    split_point  = round(split_ratio * len(data))
    train_set =   data[:split_point]
    train_key = target[:split_point]
    test_set  =   data[split_point:]
    test_key  = target[split_point:]

    learner = classifier()
    learner.train(train_set, train_key)
    prediction = learner.predict(test_set)
    return metrics.accuracy_score(test_key, prediction)

def main():
    collected_data, classifier, data_split, k_folds = parse_args()
    randomized_data = list(zip(collected_data.data, collected_data.target))
    random.shuffle(randomized_data)
    data, target = zip(*randomized_data)

    if k_folds and k_folds > 1:
        accuracy = cross_val_score(classifier, data, target, k_folds).mean()
    else:
        accuracy = train_test_score(classifier, data, target, data_split)

    percentage = int(round(100 * accuracy))
    print("The {} classifier was {}% accurate.".format(
        classifier.__name__, percentage))

if __name__ == '__main__':
    main()
