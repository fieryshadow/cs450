import sys
import random
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier

class HardCoded:
    '''A hard coded, same answer always algorithm.'''
    def __init__(self):
        self.targets = []

    def train(self, data, answers):
        self.targets = answers

    def predict(self, data, nominal=False):
        return [self.targets[0] for i in data]

class KNearestNeighbors:
    '''The k-Nearest Neighbors algorithm.'''
    def __init__(self, k):
        self.k = k
        self.targets = []

    def train(self, data, answers):
        self.targets = list(zip(data, answers))

    def predict(self, data, nominal=False):
        prediction = []
        t_data = self.targets
        for item in data:
            neighbors = []
            for t_item, t_answer in t_data:
                neighbors.append(self.d_rank(item, t_item, t_answer, nominal))
            nearest = sorted(neighbors, key=lambda x:x[0])
            possibles = [e[2] for e in nearest[:self.k]]
            prediction.append(max(possibles, key=possibles.count))
        return prediction

    def d_rank(self, item, t_item, t_answer, nominal):
        if nominal:
            rank = 20
            rank -= 3 if item[0] == t_item[0] else 0
            rank -= 4 if item[1] == t_item[1] else 0
            rank -= 1 if item[2] == t_item[2] else 0
            rank -= 1 if item[3] == t_item[3] else 0
            rank -= 2 if item[4] == t_item[4] else 0
            rank -= 5 if item[5] == t_item[5] else 0
        else:
            rank = ((item - t_item)**2).sum()
        return rank, t_item, t_answer


def parse_args():
    nominal = False
    collected_data = classifier = data_split = k_folds = None
    algorithm_matcher = { 'kNN': KNearestNeighbors, 'HC': HardCoded }
    for arg in sys.argv[1:]:
        if arg.startswith('-A='):
            algorithm = arg[3:]
            if ':' in algorithm:
                name, value = algorithm.split(':')
                classifier = algorithm_matcher[name](int(value or 3))
                clf2 = KNeighborsClassifier(n_neighbors=classifier.k)
            else:
                classifier = algorithm_matcher[algorithm]()
                clf2 = KNeighborsClassifier(n_neighbors=3)
        elif arg.startswith('-S='):
            data_split = float(arg[3:] or .7)
        elif arg.startswith('-D='):
            dataset_name = arg[3:]
            if dataset_name == 'iris':
                collected_data = datasets.load_iris()
            elif dataset_name.startswith('car:'):
                dataset = np.array(pd.io.parsers.read_csv(
                        dataset_name[4:], header=None))
                data = dataset[:,:-1]
                target = dataset[:,-1]
                names = ['buying', 'maint', 'doors',
                        'persons', 'lug_boot', 'safety']

                vals = ['low', 'med', 'high', 'vhigh', '2', '3',
                        '4', '5more', 'more', 'small', 'big']
                conv = dict(zip(vals, range(len(vals))))
                data = np.array([[conv[e] for e in row] for row in data])

                collected_data = type('CarData', (object,), {
                    'data': data, 'target': target, 'target_names': names })
                nominal = True
        elif arg.startswith('-C='):
            k_folds = int(arg[3:] or 0)

    if collected_data is None: collected_data = datasets.load_iris()
    if classifier is None: classifier = HardCoded()
    if data_split is None: data_split = .7
    return collected_data, classifier, data_split, k_folds, nominal, clf2

def cross_val_score(classifier, data, target, folds, nominal, clf2):
    k = max(0, min(folds, len(data)))
    step = len(data) // k
    results = np.zeros(k)
    results_real = np.zeros(k)
    for i, pos in zip(range(0, len(data), step), range(k)):
        train_set = data[:i] + data[i+step:]
        test_set = data[i:i+step]
        train_key = target[:i] + target[i+step:]
        test_key = target[i:i+step]

        classifier.train(train_set, train_key)
        prediction = classifier.predict(test_set, nominal)
        results[pos] = metrics.accuracy_score(test_key, prediction)

        clf2.fit(train_set, train_key)
        prediction_real = clf2.predict(test_set)
        results_real[pos] = metrics.accuracy_score(test_key, prediction_real)

    print('real', np.array(test_key))
    print('mypr', np.array(prediction))
    print('othr', np.array(prediction_real))
    return results, results_real

def train_test_score(classifier, data, target, split_ratio, nominal, clf2):
    split_point  = round(split_ratio * len(data))
    train_set =   data[:split_point]
    train_key = target[:split_point]
    test_set  =   data[split_point:]
    test_key  = target[split_point:]

    classifier.train(train_set, train_key)
    prediction = classifier.predict(test_set, nominal)

    clf2.fit(train_set, train_key)
    prediction_real = clf2.predict(test_set)

    print('real', np.array(test_key))
    print('mypr', np.array(prediction))
    print('othr', np.array(prediction_real))
    return (metrics.accuracy_score(test_key, prediction),
            metrics.accuracy_score(test_key, prediction_real))

def main():
    collected_data, classifier, data_split, k_folds, nominal, clf2 = parse_args()
    randomized_data = list(zip(collected_data.data, collected_data.target))
    random.shuffle(randomized_data)
    data, target = zip(*randomized_data)

    print() # separate displayed data from cmd prompt
    if k_folds and k_folds > 1:
        scores, scores_real = cross_val_score(
                classifier, data, target, k_folds, nominal, clf2)
        accuracy = scores.mean()
        accuracy_real = scores_real.mean()
    else:
        accuracy, accuracy_real = train_test_score(
                classifier, data, target, data_split, nominal, clf2)

    percentage = int(round(100 * accuracy))
    percentage_real = int(round(100 * accuracy_real))
    print((2*'\nThe {} classifier was {}% accurate.').format(
        classifier.__class__.__name__, percentage,
        clf2.__class__.__name__, percentage_real))

if __name__ == '__main__':
    main()
