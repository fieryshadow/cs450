print('Loading code...')
import sys
import random
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, preprocessing as prep
from sklearn.neighbors import KNeighborsClassifier

CAR = 'car'
IRIS = 'iris'
WINE = 'wine'
KNNC = 'kNN'
HCC = 'HC'
DEFAULT_NEIGHBORS = 3
CMF_ACTUAL = 'cmf_actual'
CMF_ZERO_ONE = 'cmf_zero_one'
CMF_NO_FORCE = 'cmf_no_force'
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

class MLSeed():
    def __init__(self):
        self.cross_folds = 0
        self.classifier = HardCoded()
        self.relation = NO_RELATION
        self.normalize_my_data = False
        self.normalize_their_data = False
        self.split_ratio = .6
        self.randomize_data = True
        self.randomize_splits = False
        self.confusion_matrix_format = CMF_NO_FORCE
        self.show_confusion_matrix = True
        self.show_loss_matrix = False
        self.show_accuracy = True
        self.show_sensitivity = False
        self.show_specificity = False
        self.show_precision = False
        self.show_recall = False
        self.show_f_measure = False
        self.dataset = IRIS
        self.target_pos = 0
        self.compare_classifier = KNNC


def make_sane_data(ml_seed):
    if ml_seed.dataset == IRIS:
        ml_seed.relation = LINEAR
        raw = datasets.load_iris()
        data = raw.data
        target = raw.target
        target_names = raw.target_names
        features = ['sepal length', 'sepal width', 'pedal length', 'pedal width']
        name = 'iris'
    else:
        name = ml_seed.dataset
        if name == CAR:
            ml_seed.dataset = 'http://archive.ics.uci.edu/ml/'\
                    'machine-learning-databases/car/car.data'
            ml_seed.relation = LINEAR
            ml_seed.target_pos = 6
        elif name == WINE:
            ml_seed.dataset = 'http://archive.ics.uci.edu/ml/'\
                    'machine-learning-databases/wine/wine.data'
            ml_seed.relation = LINEAR
            ml_seed.target_pos = 0
        else: name = 'loaded'
        raw = np.array(pd.io.parsers.read_csv(ml_seed.dataset, header=None))
        pos = ml_seed.target_pos
        data = np.hstack((raw[:,:pos], raw[:,pos+1:]))
        if ml_seed.relation != LINEAR:
            le = prep.LabelEncoder()
            data = np.vstack(np.transpose(
                [le.fit_transform(data[:,i]) for i in range(len(data[0]))]))
        target = raw[:,pos]
        target_names = sorted(np.unique(target))

        if name == CAR:
            features = ['buying', 'maint', 'doors',
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
        elif name == WINE:
            features = ['Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']
            target_names = [1, 2, 3]
        else:
            features = ['feature{}'.format(i) for i in range(1, len(data[0])+1)]

    if ml_seed.randomize_data:
        randomized = list(zip(data, target))
        random.shuffle(randomized)
        data, target = zip(*randomized)

    return type(name+'Data', (object,), {
        'data': np.array(data), 'target': np.array(target),
        'target_names': target_names, 'features': features })

def classify_me(ml_seed):
    if ml_seed.classifier == KNNC:
        return KNearestNeighbors(
                ml_seed.number_of_neighbors, ml_seed.normalize_my_data)
    elif ml_seed.classifier == HCC:
        return HardCoded(ml_seed.normalize_my_data)
    return HardCoded()

def classify_them(ml_seed):
    if ml_seed.compare_classifier == KNNC:
        return KNeighborsClassifier(n_neighbors=ml_seed.compare_n_neighbors)
    return None

def parse_args():
    ml_seed = MLSeed()
    for arg in sys.argv[1:]:
        if arg.startswith('-a='):
            algorithm = arg[3:]
            if ':' in algorithm:
                ml_seed.classifier, param = algorithm.split(':')
                ml_seed.number_of_neighbors = int(param or DEFAULT_NEIGHBORS)
            else: ml_seed.classifier = algorithm
        elif arg.startswith('-A='):
            algorithm = arg[3:]
            if ':' in algorithm:
                ml_seed.compare_classifier, param = algorithm.split(':')
                ml_seed.compare_n_neighbors = int(param or DEFAULT_NEIGHBORS)
            else: ml_seed.compare_classifier = algorithm
        elif arg.startswith('-s='):
            ml_seed.split_ratio = float(arg[3:] or ml_seed.split_ratio)
        elif arg.startswith('-d='):
            ml_seed.dataset = arg[3:]
        elif arg.startswith('-c='):
            ml_seed.cross_folds = int(arg[3:] or ml_seed.cross_folds)
        elif arg.startswith('-n'):
            ml_seed.normalize_my_data = True
        elif arg.startswith('-N'):
            ml_seed.normalize_their_data = True
        elif arg.startswith('-t='):
            ml_seed.target_pos = int(arg[3:] or ml_seed.target_pos)
        elif arg.startswith('-z'):
            ml_seed.confusion_matrix_format = CMF_ZERO_ONE
        elif arg.startswith('-Z'):
            ml_seed.confusion_matrix_format = CMF_ACTUAL
    return ml_seed

def make_confusion_matrix(labels, actual, prediction):
    conv = dict(zip(labels, range(len(labels))))
    matrix = np.zeros((len(labels), len(labels)))
    for real, guess in zip(actual, prediction):
        matrix[conv[real]][conv[guess]] += 1
    return type('ConfusionMatrix', (object,),
            { 'values': matrix, 'labels': labels })

def flatten_confusion_matrices(matrices):
    new_matrix = matrices[0]
    means = np.array([cm.values for cm in matrices]).mean(0)
    new_matrix.values = means
    return new_matrix

def convert_to_zero_one(matrix):
    percents = np.zeros(matrix.shape)
    s = matrix.sum()
    for i, row in enumerate(matrix):
        for j, elem in enumerate(row):
            percents[i][j] = round(elem/s, 2)
    return percents

def cross_val_score(classifier, data, target, labels, clf2, ml_seed):
    k = max(0, min(ml_seed.cross_folds, len(data)))
    step = len(data) // k
    results = np.zeros(k)
    results_real = np.zeros(k)
    confusion_matrices = []
    confusion_matrices_real = []
    for i, pos in zip(range(0, len(data), step), range(k)):
        train_set = np.vstack((data[:i], data[i+step:]))
        train_key = np.hstack((target[:i], target[i+step:]))
        test_set = data[i:i+step]
        test_key = target[i:i+step]

        classifier.train(train_set, train_key, ml_seed.relation)
        prediction = classifier.predict(test_set)
        results[pos] = metrics.accuracy_score(test_key, prediction)

        clf2.fit(train_set, train_key)
        prediction_real = clf2.predict(test_set)
        results_real[pos] = metrics.accuracy_score(test_key, prediction_real)

        confusion_matrices.append(
                make_confusion_matrix(labels, test_key, prediction))
        confusion_matrices_real.append(
                make_confusion_matrix(labels, test_key, prediction_real))

    confusion_matrix = flatten_confusion_matrices(confusion_matrices)
    confusion_matrix_real = flatten_confusion_matrices(confusion_matrices_real)
    if ml_seed.confusion_matrix_format != CMF_ACTUAL:
        confusion_matrix.values = convert_to_zero_one(confusion_matrix.values)
        confusion_matrix_real.values = convert_to_zero_one(
                confusion_matrix_real.values)
    return results, results_real, confusion_matrix, confusion_matrix_real

def train_test_score(classifier, data, target, labels, clf2, ml_seed):
    split_point = round(ml_seed.split_ratio * len(data))
    train_set =   data[:split_point]
    train_key = target[:split_point]
    test_set  =   data[split_point:]
    test_key  = target[split_point:]

    classifier.train(train_set, train_key, ml_seed.relation)
    prediction = classifier.predict(test_set)

    clf2.fit(train_set, train_key)
    prediction_real = clf2.predict(test_set)

    accuracy = metrics.accuracy_score(test_key, prediction)
    accuracy_real = metrics.accuracy_score(test_key, prediction_real)
    confusion_matrix = make_confusion_matrix(labels, test_key, prediction)
    confusion_matrix_real = \
            make_confusion_matrix(labels, test_key, prediction_real)
    return accuracy, accuracy_real, confusion_matrix, confusion_matrix_real

def main():
    print('Loading data...')
    ml_seed = parse_args()
    dataset = make_sane_data(ml_seed)
    data, target, labels = dataset.data, dataset.target, dataset.target_names
    classifier = classify_me(ml_seed)
    clf2 = classify_them(ml_seed)

    print('Processing data...')
    if ml_seed.cross_folds > 1:
        scores, scores_real, c_matrix, c_matrix_real = cross_val_score(
                classifier, data, target, labels, clf2, ml_seed)
        accuracy = scores.mean()
        accuracy_real = scores_real.mean()
    else:
        accuracy, accuracy_real, c_matrix, c_matrix_real = train_test_score(
                classifier, data, target, labels, clf2, ml_seed)

    percentage = int(round(100 * accuracy))
    percentage_real = int(round(100 * accuracy_real))
    if ml_seed.confusion_matrix_format == CMF_ZERO_ONE:
        np.set_printoptions(formatter={'float':lambda x:'{:.2f}'.format(x)[1:]})
        c_matrix.values = convert_to_zero_one(c_matrix.values)
        c_matrix_real.values = convert_to_zero_one(c_matrix_real.values)
    else:
        np.set_printoptions(formatter={'float':lambda x:'{:>7.2f}'.format(x)})
    print((2*('\nThe {} classifier was {}% accurate,'
        ' with confusion matrix:\n{}\n') +
        '\nConfusion matrix class order:\n{}').format(
        classifier.__class__.__name__, percentage, c_matrix.values,
        clf2.__class__.__name__, percentage_real, c_matrix_real.values,
        c_matrix.labels))

if __name__ == '__main__':
    main()
