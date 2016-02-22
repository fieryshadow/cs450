print('Loading code...', end='\r')
import numpy as np
from sys import argv as sys_argv
from pandas.io.parsers import read_csv as load_dataset
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score as calc_accuracy
from sklearn.metrics.cluster import entropy as calc_entropy

IRIS = 'iris'
CAR = 'car'
WINE = 'wine'
LENSES = 'lenses'
VOTING = 'voting'
CREDIT = 'credit'
CHESS = 'chess'
PIMA = 'pima'
KNNC = 'kNN'
ID3C = 'ID3'
NNC = 'NN'
HCC = 'HC'
DEFAULT_LEARNING_RATE = .1
DEFAULT_NUM_EPOCHS = 77
DEFAULT_MOMENTUM = .2
DEFAULT_NEIGHBORS = 3
CMF_ACTUAL = 'cmf_actual'
CMF_ZERO_ONE = 'cmf_zero_one'
CMF_NO_FORCE = 'cmf_no_force'
NO_RELATION = 0
LINEAR = 1
DATASET_NAMES = [CAR, WINE, LENSES, VOTING, CREDIT, CHESS, PIMA]
RELATIONS = dict(zip(DATASET_NAMES, 2*[LINEAR] + 5*[NO_RELATION]))
TARGET_COLS = dict(zip(DATASET_NAMES, [6, 0, 5, 0, 15, 6, 8]))
WEBSITES = dict(zip(DATASET_NAMES, map(
    lambda x:'https://archive.ics.uci.edu/ml/machine-learning-databases'+x, '''
/car/car.data
/wine/wine.data
/lenses/lenses.data
/voting-records/house-votes-84.data
/credit-screening/crx.data
/chess/king-rook-vs-king/krkopt.data
/pima-indians-diabetes/pima-indians-diabetes.data
'''.strip().split('\n'))))

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
    def __init__(self, k):
        self.k = k
        self.targets = []
        self.relation = NO_RELATION

    def train(self, data, answers, relation=NO_RELATION):
        self.relation = relation
        self.targets = list(zip(data, answers))

    def predict(self, data):
        d_rank = self.rank_l if self.relation == LINEAR else self.rank_n
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
        dist = ((item - t_item) ** 2).sum()
        return dist, t_item, t_answer

    def rank_n(self, item, t_item, t_answer):
        rank = len(item) - (item == t_item).sum()
        return rank, t_item, t_answer

class ID3Node():
    def __init__(self):
        self.default = None
        self.feature = None
        self.mapping = {}

class ID3Tree():
    def __init__(self):
        self.tree = ID3Node()

    def train(self, data, answers, relation=NO_RELATION):
        self.tree = ID3Node()
        self._build(data, answers, [], self.tree)

    def _build(self, data, answers, used, node):
        if len(np.unique(answers)) == 1 or len(used) == len(data[0]):
            node.mapping = max(answers, key=answers.count)
            return
        min_entropy = None
        min_feature = None
        min_mapping = None
        for feature in range(len(data[0])):
            if feature in used: continue
            entropy, mapping = self.entropy_map(feature, data, answers)
            if min_entropy is None or entropy < min_entropy:
                min_entropy = entropy
                min_feature = feature
                min_mapping = mapping
        temp = list(answers)
        node.default = max(temp, key=temp.count)
        node.feature = min_feature
        for feature_value, dataset_subset in min_mapping.items():
            data_subset, target_subset = zip(*dataset_subset)
            node.mapping[feature_value] = ID3Node()
            self._build(data_subset, target_subset,
                    used+[min_feature], node.mapping[feature_value])

    def entropy_map(self, feature, dataset, targets):
        mapping = {}
        for data, target in zip(dataset, targets):
            if data[feature] not in mapping.keys():
                mapping[data[feature]] = []
            mapping[data[feature]].append((data, target))
        entropy = sum([calc_entropy([t for d, t in s]) * len(s) / len(dataset)
                for s in mapping.values()])
        return entropy, mapping

    def predict(self, dataset):
        prediction = []
        for data in dataset:
            node = self.tree
            while isinstance(node.mapping, dict):
                if data[node.feature] not in node.mapping.keys():
                    node = type('node', (object,), { 'mapping': node.default })
                    break
                node = node.mapping[data[node.feature]]
            prediction.append(node.mapping)
        return prediction

class Neuron():
    def __init__(self, num_in, batch):
        self.weights = np.array(
                [np.random.uniform(-1, 1) for i in range(num_in + 1)])
        self.prev_delta = np.zeros(num_in + 1)
        if batch: self.prev_delta = [self.prev_delta]
        self.activation = 0
        self.error = 0

    def get_output(self, inputs):
        weighted_sum = self.weights[0] + (self.weights[1:] * inputs).sum()
        self.activation = 1 / (1 + np.exp(-weighted_sum))
        return self.activation

class NeuralLayer():
    def __init__(self, num_in, num_out, batch):
        self.neurons = np.array([Neuron(num_in, batch) for i in range(num_out)])
        self.inputs = [1] # bias input
        self.batch = batch

    def get_output(self, inputs):
        self.inputs[1:] = inputs # update non bias inputs
        return np.array([neuron.get_output(inputs) for neuron in self.neurons])

    def calculate_error(self, nlayer):
        hidden = isinstance(nlayer, NeuralLayer)
        for i, neuron in enumerate(self.neurons):
            third = neuron.activation - nlayer[i] if not hidden else \
                    sum(n.error * n.weights[i+1] for n in nlayer.neurons)
            neuron.error = neuron.activation * (1 - neuron.activation) * third

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            rne = learning_rate * neuron.error
            delta = np.array([rne * activation for activation in self.inputs])
            if self.batch:
                delta += momentum * neuron.prev_delta[-1]
                neuron.prev_delta.append(-delta)
            else:
                delta += momentum * neuron.prev_delta
                neuron.prev_delta = -delta
                neuron.weights -= delta

    def batch_complete(self):
        for neuron in self.neurons:
            neuron.weights += np.array(neuron.prev_delta).sum(axis=0)
            neuron.prev_delta = [np.zeros(len(neuron.weights))]

class NeuralNetwork():
    def __init__(self, ml_seed, hidden=None, num_epochs=None,
            momentum=None, learning_rate=None):
        self.batch = ml_seed.batch_processing
        self.show_epoch_num = ml_seed.show_epoch_num
        self.show_epoch_stats = ml_seed.show_epoch_accuracy
        self.learning_rate = learning_rate or DEFAULT_LEARNING_RATE
        self.num_epochs = num_epochs or DEFAULT_NUM_EPOCHS
        self.momentum = momentum or DEFAULT_MOMENTUM
        self.hidden = hidden or []
        self.target_names = []
        self.layers = []

    def make_layers(self, start_num, end_num):
        self.layers = []
        prev_amount = start_num
        for amount in self.hidden:
            self.layers.append(NeuralLayer(prev_amount, amount, self.batch))
            prev_amount = amount
        self.layers.append(NeuralLayer(prev_amount, end_num, self.batch))

    def train(self, data, targets, relation=NO_RELATION):
        self.target_names = np.unique(targets)
        self.make_layers(len(data[0]), len(self.target_names))

        accuracy = []
        dt = np.array(list(zip(data, targets)))
        for i in range(self.num_epochs):
            np.random.shuffle(dt) # make sure NN doesn't learn feed ordering
            if self.show_epoch_num:
                print('In epoch {}/{}'.format(i+1, self.num_epochs), end='\r')
            accuracy.append([])
            for d, t in dt:
                real_out = self.get_output(d)
                nlayer = (self.target_names == t).astype(int) # ideal outputs
                accuracy[-1].append(
                        (np.argmax(real_out) == np.argmax(nlayer)) * 1)
                for layer in reversed(self.layers):
                    layer.calculate_error(nlayer)
                    nlayer = layer
                for layer in reversed(self.layers):
                    layer.update_weights(self.learning_rate, self.momentum)
                if self.batch:
                    for layer in self.layers: layer.batch_complete()
            accuracy[-1] = sum(accuracy[-1]) / len(accuracy[-1])
        if self.show_epoch_num:
            print('Finished processing', self.num_epochs, 'epochs')
        if self.show_epoch_stats:
            print('\nAccuracies for each epoch of the most recent training:')
            np.set_printoptions(
                    formatter={'float':lambda x:'{:.3f}'.format(x)[1:]})
            print(np.array(accuracy))

    def get_output(self, data):
        for layer in self.layers:
            data = layer.get_output(data)
        return data

    def predict(self, data):
        return [self.target_names[np.argmax(self.get_output(d))] for d in data]

class MLSeed():
    def __init__(self):
        self.cross_folds = 0
        self.classifier = HardCoded()
        self.relation = NO_RELATION
        self.normalize_my_data = False
        self.normalize_their_data = False
        self.batch_processing = False
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
        self.show_tree = False
        self.show_epoch_accuracy = False
        self.show_epoch_num = True
        self.dataset = IRIS
        self.target_pos = 0
        self.compare_classifier = KNNC
        self.clusterize = 0


def make_sane_data(ml_seed):
    if ml_seed.dataset == IRIS:
        ml_seed.relation = LINEAR
        raw = load_iris()
        data = raw.data
        target_names = raw.target_names
        target = [target_names[i] for i in raw.target]
        features = ['sepalL', 'sepalW', 'pedalL', 'pedalW']
        name = 'iris'
    else:
        name = ml_seed.dataset
        sep = '\s+' if name == LENSES else ','
        if name in DATASET_NAMES:
            ml_seed.dataset = WEBSITES[name]
            ml_seed.relation = RELATIONS[name]
            ml_seed.target_pos = TARGET_COLS[name]
        else: name = 'loaded'
        raw = np.array(load_dataset(ml_seed.dataset, sep=sep, header=None))
        pos = ml_seed.target_pos
        data = np.hstack((raw[:,:pos], raw[:,pos+1:]))
        if ml_seed.relation != LINEAR:
            le = LabelEncoder()
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
            data = np.array(
                    [[conv[i][e] for i, e in enumerate(row)] for row in data])
            target_names = ['unacc', 'acc', 'good', 'vgood']
        elif name == WINE:
            features = ['Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']
            target_names = [1, 2, 3]
        else:
            features = ['feature' + str(i) for i in range(1, len(data[0]) + 1)]

    if ml_seed.randomize_data:
        data, target = zip(*np.random.permutation(list(zip(data, target))))

    if ml_seed.clusterize:
        data = np.array(data)
        km = KMeans(n_clusters=ml_seed.clusterize)
        data = np.vstack(np.transpose([km.fit_predict(
            data[:,i].reshape(-1, 1)) for i in range(len(data[0]))]))

    return type(name+'Data', (object,), {
        'data': np.array(data), 'target': np.array(target),
        'target_names': target_names, 'features': features })

def classify_me(ml_seed):
    if ml_seed.classifier.startswith(KNNC):
        _, param = ml_seed.classifier.split(':')
        k = int(param or DEFAULT_NEIGHBORS)
        return KNearestNeighbors(k)
    elif ml_seed.classifier == ID3C:
        return ID3Tree()
    elif ml_seed.classifier.startswith(NNC):
        params = ml_seed.classifier.split(':')
        length = len(params)
        hidden = map(int, params[1].split(',')) if length > 1 else []
        epochs = int(params[2]) if length > 2 else DEFAULT_NUM_EPOCHS
        momentum = float(params[3]) if length > 3 else DEFAULT_MOMENTUM
        learn_rate = float(params[4]) if length > 4 else DEFAULT_LEARNING_RATE
        return NeuralNetwork(ml_seed, hidden, epochs, momentum, learn_rate)
    elif ml_seed.classifier == HCC:
        return HardCoded()
    return HardCoded()

def classify_them(ml_seed):
    if ml_seed.compare_classifier.startswith(KNNC):
        _, param = ml_seed.compare_classifier.split(':')
        k = int(param or DEFAULT_NEIGHBORS)
        return KNeighborsClassifier(n_neighbors=k)
    elif ml_seed.compare_classifier == ID3C:
        return DecisionTreeClassifier()
    return None

def parse_args():
    ml_seed = MLSeed()
    for arg in sys_argv[1:]:
        if arg.startswith('-a='):
            ml_seed.classifier = arg[3:]
        elif arg.startswith('-A='):
            ml_seed.compare_classifier = arg[3:]
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
        elif arg.startswith('-Y'):
            ml_seed.show_tree = True
        elif arg.startswith('-C='):
            ml_seed.clusterize = int(arg[3:] or ml_seed.clusterize)
        elif arg.startswith('-e'):
            ml_seed.show_epoch_accuracy = True
        elif arg.startswith('-E'):
            ml_seed.show_epoch_num = False
        elif arg.startswith('-b'):
            ml_seed.batch_processing = True
    return ml_seed

def display_tree(tree, level=0, pre='->'):
    if isinstance(tree.mapping, dict):
        print('  '*level + pre + repr(tree.feature) + ':')
        for feature_value, child in tree.mapping.items():
            display_tree(child, level+1, repr(feature_value)+'->')
    else: print('  '*level + pre + repr(tree.mapping))

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
            percents[i][j] = round(elem / s, 2)
    return percents

def make_normalizer(data):
    data = np.array(data)
    normalizer = MinMaxScaler().fit(data.astype(float))
    return normalizer.transform

def cross_val_score(classifier, data, target, labels, clf2, ml_seed):
    k = max(0, min(ml_seed.cross_folds, len(data)))
    step = len(data) // k
    results = np.zeros(k)
    results_real = np.zeros(k)
    confusion_matrices = []
    confusion_matrices_real = []
    normalizer = make_normalizer(data)
    for i, pos in zip(range(0, len(data), step), range(k)):
        train_set = np.vstack((data[:i], data[i+step:]))
        train_key = np.hstack((target[:i], target[i+step:]))
        test_set = data[i:i+step]
        test_key = target[i:i+step]

        if ml_seed.normalize_my_data:
            my_train_set = normalizer(train_set)
            my_test_set = normalizer(test_set)
        else:
            my_train_set = train_set
            my_test_set = test_set
        classifier.train(my_train_set, train_key, ml_seed.relation)
        prediction = classifier.predict(my_test_set)
        results[pos] = calc_accuracy(test_key, prediction)

        if ml_seed.normalize_their_data:
            their_train_set = normalizer(train_set)
            their_test_set = normalizer(test_set)
        else:
            their_train_set = train_set
            their_test_set = test_set
        clf2.fit(their_train_set, train_key)
        prediction_real = clf2.predict(their_test_set)
        results_real[pos] = calc_accuracy(test_key, prediction_real)

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
    normalizer = make_normalizer(data)
    split_point = round(ml_seed.split_ratio * len(data))
    train_set =   data[:split_point]
    train_key = target[:split_point]
    test_set  =   data[split_point:]
    test_key  = target[split_point:]

    if ml_seed.normalize_my_data:
        my_train_set = normalizer(train_set)
        my_test_set = normalizer(test_set)
    else:
        my_train_set = train_set
        my_test_set = test_set
    classifier.train(my_train_set, train_key, ml_seed.relation)
    prediction = classifier.predict(my_test_set)

    if ml_seed.normalize_their_data:
        their_train_set = normalizer(train_set)
        their_test_set = normalizer(test_set)
    else:
        their_train_set = train_set
        their_test_set = test_set
    clf2.fit(their_train_set, train_key)
    prediction_real = clf2.predict(their_test_set)

    accuracy = calc_accuracy(test_key, prediction)
    accuracy_real = calc_accuracy(test_key, prediction_real)
    confusion_matrix = make_confusion_matrix(labels, test_key, prediction)
    confusion_matrix_real = make_confusion_matrix(
            labels, test_key, prediction_real)
    return accuracy, accuracy_real, confusion_matrix, confusion_matrix_real

def main():
    print('Retrieving data...', end='\r')
    ml_seed = parse_args()
    dataset = make_sane_data(ml_seed)
    data, target, labels = dataset.data, dataset.target, dataset.target_names
    classifier = classify_me(ml_seed)
    clf2 = classify_them(ml_seed)

    print('Learning structures...')
    if ml_seed.cross_folds > 1:
        scores, scores_real, c_matrix, c_matrix_real = cross_val_score(
                classifier, data, target, labels, clf2, ml_seed)
        accuracy = scores.mean()
        accuracy_real = scores_real.mean()
    else:
        accuracy, accuracy_real, c_matrix, c_matrix_real = train_test_score(
                classifier, data, target, labels, clf2, ml_seed)

    if ml_seed.show_tree:
        print() # give me some space!
        display_tree(classifier.tree)

    percentage = int(round(100 * accuracy))
    percentage_real = int(round(100 * accuracy_real))
    if ml_seed.confusion_matrix_format == CMF_ZERO_ONE:
        np.set_printoptions(formatter={
            'float':lambda x:'{:.2f}'.format(x)[1:] if x else '.__'})
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
