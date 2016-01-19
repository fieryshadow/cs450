print('Loading code...')
import sys
import random
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, preprocessing as prep
from sklearn.neighbors import KNeighborsClassifier

NO_RELATION = 0
LINEAR = 1

class HardCoded:
    '''A hard coded, same answer always algorithm.'''
    def __init__(self):
        self.targets = []

    def train(self, data, answers, relation=NO_RELATION):
        self.targets = answers

    def predict(self, data):
        return [self.targets[0] for i in data]

class KNearestNeighbors:
    '''The k-Nearest Neighbors algorithm.'''
    def __init__(self, k, normalize=True):
        self.k = k
        self.targets = []
        self.relation = NO_RELATION
        self.normalize = normalize
        self.normalizer = type(
                'notransform', (object,), { 'transform': lambda x:x })

    def train(self, data, answers, relation=NO_RELATION):
        self.relation = relation
        data = np.array(data)
        if self.normalize and relation == LINEAR:
            self.normalizer = prep.MinMaxScaler().fit(data.astype(float))
            data = self.normalizer.transform(data)
        self.targets = list(zip(data, answers))

    def predict(self, data):
        d_rank = self.rank_l if self.relation == LINEAR else self.rank_n
        if self.normalize and self.relation == LINEAR:
            data = self.normalizer.transform(data)
        prediction = []
        t_data = self.targets
        for item in data:
            neighbors = []
            for t_item, t_answer in t_data:
                neighbors.append(d_rank(item, t_item, t_answer))
            nearest = sorted(neighbors, key=lambda x:x[0])
            possibles = [e[2] for e in nearest[:self.k]]
            prediction.append(max(possibles, key=possibles.count))
        return prediction

    def rank_l(self, item, t_item, t_answer):
        dist = ((item - t_item)**2).sum()
        return dist, t_item, t_answer

    def rank_n(self, item, t_item, t_answer):
        rank = len(item) - (item == t_item).sum()
        return rank, t_item, t_answer


def parse_args():
    relation = NO_RELATION
    collected = classifier = data_split = k_folds = normalize = None
    algorithm_matcher = { 'kNN': KNearestNeighbors, 'HC': HardCoded }
    args = sys.argv[1:]
    for arg in args:
        if arg == '-n': normalize = True
    if normalize is None: normalize = False

    for arg in args:
        if arg.startswith('-a='):
            algorithm = arg[3:]
            if ':' in algorithm:
                name, value = algorithm.split(':')
                classifier = algorithm_matcher[name](int(value or 3), normalize)
                clf2 = KNeighborsClassifier(n_neighbors=classifier.k)
            else:
                classifier = algorithm_matcher[algorithm]()
                clf2 = KNeighborsClassifier(n_neighbors=3)
        elif arg.startswith('-s='):
            data_split = float(arg[3:] or .7)
        elif arg.startswith('-d='):
            dataset_name = arg[3:]
            if dataset_name == 'iris':
                collected = datasets.load_iris()
                relation = LINEAR
            else:
                name, loc = dataset_name.split(':', 1)
                dataset = np.array(pd.io.parsers.read_csv(loc, header=None))
                if name == 'car':
                    data = dataset[:,:-1]
                    target = dataset[:,-1]
                    attrbutes = ['buying', 'maint', 'doors',
                            'persons', 'lug_boot', 'safety']
                    conv = [{ 'low': 0, 'med': 1, 'high': 2, 'vhigh': 3 },
                            { 'low': 0, 'med': 1, 'high': 2, 'vhigh': 3 },
                            { '2': 0, '3': 1, '4': 2, '5more': 3 },
                            { '2': 0, '4': 1, 'more': 2 },
                            { 'small': 0, 'med': 1, 'big': 2 },
                            { 'low': 0, 'med': 1, 'high': 2 }]
                    data = np.array([[conv[i][e] for i, e in enumerate(row)]
                        for row in data])
                    target_names = ['unacc', 'acc', 'good', 'vgood']
                elif name == 'wine':
                    data = dataset[:,1:]
                    target = dataset[:,0]
                    attrbutes = ['Alcohol', 'Malic acid', 'Ash',
                            'Alcalinity of ash', 'Magnesium', 'Total phenols',
                            'Flavanoids', 'Nonflavanoid phenols',
                            'Proanthocyanins', 'Color intensity', 'Hue',
                            'OD280/OD315 of diluted wines', 'Proline']
                    target_names = [1, 2, 3]
                else: continue
                collected = type(name+'Data', (object,), {
                    'data': data, 'target': target,
                    'target_names': target_names, 'attr_names': attrbutes })
                relation = LINEAR
        elif arg.startswith('-c='):
            k_folds = int(arg[3:] or 0)

    if collected is None: collected = datasets.load_iris()
    if classifier is None: classifier = HardCoded()
    if data_split is None: data_split = .7
    return collected, classifier, data_split, k_folds, relation, clf2

def cross_val_score(classifier, data, target, folds, relation, clf2):
    k = max(0, min(folds, len(data)))
    step = len(data) // k
    results = np.zeros(k)
    results_real = np.zeros(k)
    for i, pos in zip(range(0, len(data), step), range(k)):
        train_set = data[:i] + data[i+step:]
        test_set = data[i:i+step]
        train_key = target[:i] + target[i+step:]
        test_key = target[i:i+step]

        classifier.train(train_set, train_key, relation)
        prediction = classifier.predict(test_set)
        results[pos] = metrics.accuracy_score(test_key, prediction)

        clf2.fit(train_set, train_key)
        prediction_real = clf2.predict(test_set)
        results_real[pos] = metrics.accuracy_score(test_key, prediction_real)

    print('real', np.array(test_key))
    print('mypr', np.array(prediction))
    print('othr', np.array(prediction_real))
    return results, results_real

def train_test_score(classifier, data, target, split_ratio, relation, clf2):
    split_point = round(split_ratio * len(data))
    train_set =   data[:split_point]
    train_key = target[:split_point]
    test_set  =   data[split_point:]
    test_key  = target[split_point:]

    classifier.train(train_set, train_key, relation)
    prediction = classifier.predict(test_set)

    clf2.fit(train_set, train_key)
    prediction_real = clf2.predict(test_set)

    print('real', np.array(test_key))
    print('mypr', np.array(prediction))
    print('othr', np.array(prediction_real))
    return (metrics.accuracy_score(test_key, prediction),
            metrics.accuracy_score(test_key, prediction_real))

def main():
    print('Loading data...')
    (collected_data, classifier, data_split,
            k_folds, relation, clf2) = parse_args()
    randomized_data = list(zip(collected_data.data, collected_data.target))
    random.shuffle(randomized_data)
    data, target = zip(*randomized_data)

    print('Processing data...\n')
    if k_folds and k_folds > 1:
        scores, scores_real = cross_val_score(
                classifier, data, target, k_folds, relation, clf2)
        accuracy = scores.mean()
        accuracy_real = scores_real.mean()
    else:
        accuracy, accuracy_real = train_test_score(
                classifier, data, target, data_split, relation, clf2)

    percentage = int(round(100 * accuracy))
    percentage_real = int(round(100 * accuracy_real))
    print((2*'\nThe {} classifier was {}% accurate.').format(
        classifier.__class__.__name__, percentage,
        clf2.__class__.__name__, percentage_real))

if __name__ == '__main__':
    main()
