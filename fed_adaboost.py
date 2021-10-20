# -*- coding: utf-8 -*-
from __future__ import division, print_function, annotations
import os
from typing import Any, Tuple, Dict, List, Generator
from copy import deepcopy
from math import log
import tqdm
import pandas as pd
import numpy as np
from numpy.random import choice, permutation
from sklearn import datasets
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from optparse import OptionParser
import dload
import urllib.request as ureq
import json
import gzip
import wandb


def manage_options() -> Dict[str, Any]:
    parser = OptionParser(usage="usage: %prog [options] dataset",
                          version="%prog 0.1",
                          description="Testing Distboost and Preweak from Cooper et al. 2017. "\
                          "Supported dataset: breast, mnistXY, sonar, ionospehere, banknote, "\
                          "spambase, particle, and forestcover12. Note that mnistXY is a binary "\
                          "classification subset of MNIST in which X is a digit and Y is another digit with "\
                          "X,Y in {0, 1, ..., 9}. The `test_size` argument is ignored for the mnist, particle and "\
                          "the forestcover12 datasets for which it is used a default split size. The dataset "\
                          "can be also a path to a svmlight file.")
    parser.add_option("-s", "--seed",
                      dest="seed", default=42, type="int", 
                      help="Pseudo-random seed for replicability purposes - default=42")
    parser.add_option("-t", "--test_size",
                      dest="test_size", default=.1, type="float",
                      help="Test set size in percentage (0,1) - default=0.1")
    parser.add_option("-n", "--nclients",
                      dest="n_clients", default=10, type="int",
                      help="Number of clients (>= 1) - default=10")
    parser.add_option("-m", "--model",
                      dest="model", default="adaboost", type="str",
                      help="The model to train and test. Supported models: adaboost, my_ada, "\
                           "distboost and preweak - default=adaboost")
    parser.add_option("-z", "--normalize",
                      dest="normalize", default=False, action="store_true",
                      help="Whether the instances has to be normalized or not - default=False")
    parser.add_option("-l", "--labels",
                      dest="tags", default="", type="str",
                      help="list of comma separated tags to be added in the wandb lod - default=''")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print("Dataset argument missing. Please check the documentation: `python fed_adaboost.py -h`")
        exit(0)
    options.dataset = args[0]
    return options


def load_covtype_dataset(seed: int=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile("covtype.data"):
        dload.save_unzip("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz")
    covtype_df = pd.read_csv("covtype.data", header=None)
    covtype_df = covtype_df[covtype_df[54] < 3]
    X = covtype_df.loc[:, :53].to_numpy()
    y = (2*(covtype_df.loc[:, 54] - 1) - 1).to_numpy()
    np.random.seed(seed=seed)
    ids = permutation(X.shape[0])
    X, y = X[ids], y[ids]
    X_train, X_test = X[:250000], X[250000:]
    y_train, y_test = y[:250000], y[250000:]
    return X_train, X_test, y_train, y_test


def load_particle_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile("MiniBooNE_PID.txt"):
        ureq.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt", "MiniBooNE_PID.txt")

    with open("MiniBooNE_PID.txt") as f:
        cnt_pos, cnt_neg = map(int, f.readline().strip().split(" "))
        X = []
        for _ in range(cnt_pos + cnt_neg):
            line = f.readline().strip().replace(" -", "  -").split("  ")
            X.append(list(map(float, line)))
        y = np.concatenate([np.full((cnt_pos,), 1), np.full((cnt_neg,), -1)])
    X = np.array(X)
    ids = list(np.random.permutation(cnt_pos + cnt_neg))
    X, y = X[ids], y[ids]
    X_train, X_test = X[:80000], X[80000:]
    y_train, y_test = y[:80000], y[80000:]
    return X_train, X_test, y_train, y_test

BASE_URL_MNIST: str = "http://yann.lecun.com/exdb/mnist/"
URL_MNIST: Dict[str, Tuple[str, str]] = {
    "training_images": (BASE_URL_MNIST, "train-images-idx3-ubyte.gz"),
    "test_images": (BASE_URL_MNIST, "t10k-images-idx3-ubyte.gz"),
    "training_labels": (BASE_URL_MNIST, "train-labels-idx1-ubyte.gz"),
    "test_labels": (BASE_URL_MNIST, "t10k-labels-idx1-ubyte.gz")
}

def load_mnist_dataset():
    print("Downloading MNIST")
    for k, v in tqdm.tqdm(URL_MNIST.items()):
        ureq.urlretrieve(v[0] + v[1], v[1])
    
    mnist = {}
    for k in ["training_images", "test_images"]:
        name = URL_MNIST[k][1]
        with gzip.open(name, 'rb') as f:
            mnist[k] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for k in ["training_labels", "test_labels"]:
        name = URL_MNIST[k][1]
        with gzip.open(name, 'rb') as f:
            mnist[k] = np.frombuffer(f.read(), np.uint8, offset=8)

    return mnist["training_images"], \
           mnist["test_images"], \
           mnist["training_labels"], \
           mnist["test_labels"]


UCI_BASE_URL : str = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

UCI_URL_AND_CLASS : Dict[str, Tuple[str, int]] = {
    "spambase" : (UCI_BASE_URL + "spambase/spambase.data", 57),
    "sonar" : (UCI_BASE_URL + "undocumented/connectionist-bench/sonar/sonar.all-data", 60),
    "ionosphere" : (UCI_BASE_URL + "ionosphere/ionosphere.data", 34),
    "banknote" : (UCI_BASE_URL + "00267/data_banknote_authentication.txt", 4)
}

def load_binary_classification_dataset(name_or_path: str,
                                       normalize: bool=True,
                                       test_size: float=0.1, #real range (0,1)
                                       seed: int=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a UCI binary classification dataset. `name_or_path` must be a valid dataset name
    (string) or a path to a svmlight file. The supported dataset names are: breast, mnistXY, sonar,
    ionospehere, banknote, spambase, particle, and forestcover12. Note that mnistXY is a binary
    classification subset of MNIST in which X is a digit and Y is another digit with
    X,Y in {0, 1, ..., 9}. The `test_size` argument is ignored for the mnist, particle and
    the forestcover12 datasets.
    """
    if name_or_path == "breast":
        dataset = datasets.load_breast_cancer()
        X, y = dataset.data, dataset.target
    # Expected: 
    elif name_or_path.startswith("mnist"):
        digits = name_or_path.replace("mnist", "")
        d1, d2 = int(digits[0]), int(digits[1])
        dataset = datasets.load_digits()
        X_train, X_test, y_train, y_test = load_mnist_dataset()
        ids_tr = np.concatenate([np.where(y_train == d1)[0].flatten(),
                                 np.where(y_train == d2)[0].flatten()])
        ids_te = np.concatenate([np.where(y_test == d1)[0].flatten(),
                                 np.where(y_test == d2)[0].flatten()])
        X_train = X_train[ids_tr, :]
        y_train = y_train[ids_tr].astype(int)
        X_test = X_test[ids_te, :]
        y_test = y_test[ids_te].astype(int)
        y_train[np.where(y_train == d1)] = -1
        y_train[np.where(y_train == d2)] = 1
        y_test[np.where(y_test == d1)] = -1
        y_test[np.where(y_test == d2)] = 1
        return X_train, X_test, y_train, y_test
    elif name_or_path in {"sonar", "ionosphere", "banknote", "spambase"}:
        url, label_id = UCI_URL_AND_CLASS[name_or_path]
        data = pd.read_csv(url, header=None).to_numpy()
        y = LabelEncoder().fit_transform(data[:, label_id])
        X = np.delete(data, [label_id], axis=1).astype('float64')
    elif name_or_path == "particle":
        # test_size is ignored for this dataset
        return load_particle_dataset()
    elif name_or_path == "forestcover12":
        # test_size is ignored for this dataset
        return load_covtype_dataset(seed)
    else:
        X, y = load_svmlight_file(name_or_path)
        y = LabelEncoder().fit_transform(y)
        X = X.toarray()

    y = 2*y - 1 # 0/1 labels to -1/1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if normalize:
        scaler = StandardScaler().fit(X_train)   
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Given the number of clients, the splitting is deterministic.
# However, if the dataset has been shuffled it is ok.
def split_dataset(X: np.ndarray,
                  y: np.ndarray,
                  n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n > X.shape[0]:
        raise ValueError("# of users must be <= than the # of examples in the training set.")
    
    tot = X.shape[0]
    X_tr = [X[list(range(i, tot, n)), :] for i in range(n)]
    y_tr = [y[list(range(i, tot, n))] for i in range(n)]
    return X_tr, y_tr


def split_data_powerlaw(X: np.ndarray,
                        y: np.ndarray,
                        n: int,
                        mn: int=2,
                        p: float=4.) -> Tuple[np.ndarray, np.ndarray]:
    assert mn*n <= X.shape[0], "# of instances must be > than mn*n"
    s = np.array(np.random.power(p, X.shape[0] - mn*n) * n, dtype=int)
    m = np.array([[i]*mn for i in range(n)]).flatten()
    assignment = np.concatenate([s, m])
    np.random.shuffle(assignment)
    X_tr, y_tr = [], []
    for i in range(n):
        idx = np.where(assignment == i)[0]
        X_tr.append(X[idx])
        y_tr.append(y[idx])
    return X_tr, y_tr


class Boosting():
    def __init__(self: Boosting,
                 n_clf: int=10,
                 clf_class: ClassifierMixin=DecisionTreeClassifier()):
        self.n_clf = n_clf
        self.clf_class = clf_class
        self.clfs = []
        self.alpha = []
    
    def fit(self: Boosting,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: List[int],
            seed: int=42) -> Generator[Boosting]:
        raise NotImplementedError()
    
    def num_weak_learners(self: Boosting):
        return len(self.clfs)
    
    def predict(self: Boosting,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros(np.shape(X)[0])
        for i, clf in enumerate(self.clfs):
            y_pred += self.alpha[i] * clf.predict(X)
        return 2*(y_pred.flatten() >= 0).astype(int) - 1 


class Adaboost(Boosting):

    def fit(self: Adaboost,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: List[int],
            seed: int=42) -> Generator[Adaboost]:

        np.random.seed(seed)
        cks = set(checkpoints)
        n_samples = X.shape[0]
        D = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            clf = deepcopy(self.clf_class)
            ids = choice(n_samples, size=n_samples, replace=True, p=D)
            X_, y_ = X[ids], y[ids]
            clf.fit(X_, y_)

            predictions = clf.predict(X)
            min_error = sum(D[y != predictions])
            self.alpha.append(0.5 * log((1.0 - min_error) / (min_error + 1e-10))) # kind of additive smoothing
            D *= np.exp(-self.alpha[t] * y * predictions)
            D /= np.sum(D)
            self.clfs.append(clf)

            if (t+1) in cks:
                yield self


# Support class for Distboost
class Hyp():
    def __init__(self: Hyp,
                 ht: List[ClassifierMixin]):
        self.ht = ht
    
    def predict(self: Hyp,
                X: np.ndarray) -> np.ndarray:
        return np.sign(np.sum([h.predict(X) for h in self.ht], axis=0))


class Distboost(Boosting):

    def fit(self: Distboost,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: List[int],
            seed: int=42) -> Generator[Distboost]:

        np.random.seed(seed)
        cks = set(checkpoints)
        n_samples = sum([x.shape[0] for x in X])
        D = [np.full(x.shape[0], (1 / n_samples)) for x in X]
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            ht = []
            for j, X_ in enumerate(X):
                clf = deepcopy(self.clf_class)
                ids = choice(X_.shape[0], size=X_.shape[0], replace=True, p=D[j]/np.sum(D[j]))
                X__, y__ = X_[ids], y[j][ids]
                clf.fit(X__, y__)
                ht.append(clf)

            H = Hyp(ht)
            self.clfs.append(H)
            
            min_error = 0
            predictions = []
            for j, X_ in enumerate(X):
                predictions.append(H.predict(X_))
                min_error += sum(D[j][y[j] != predictions[j]])
            self.alpha.append(0.5 * log((1.0 - min_error) / (min_error + 1e-10)))
            
            for j, X_ in enumerate(X):
                D[j] *= np.exp(-self.alpha[t] * y[j] * predictions[j])
            Dsum = sum([np.sum(d) for d in D])
            
            for d in D:
                d /= Dsum

            if (t+1) in cks:
                yield self


class Preweak(Boosting):
    
    def fit(self: Preweak,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: List[int],
            seed: int=42) -> Generator[Preweak]:
        
        np.random.seed(seed)
        cks = set(checkpoints)
        ht = []
        for j, X_ in enumerate(X):
            clf = Adaboost(self.n_clf, self.clf_class)
            for h in clf.fit(X_, y[j], checkpoints):
                ht.append(h)

        # merge the datasets into one (not possible in a real distributed/federated scenario)
        X_ = np.vstack(X) 
        y_ = np.concatenate(y)
        
        # precompute the predictions so then I simply have to draw according to the sampling 
        ht_pred = {h : h.predict(X_) for h in ht}
        n_samples = X_.shape[0]
        D = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            ids = choice(X_.shape[0], size=X_.shape[0], replace=True, p=D)
            y__ = y_[ids]

            min_error = 1000
            top_model = None
            for h, hpred in ht_pred.items():
                err = sum(D[y__ != hpred[ids]])
                if err < min_error:
                    top_model = h
                    min_error = err

            self.alpha.append(0.5 * log((1.0 - min_error) / (min_error + 1e-10)))
            D *= np.exp(-self.alpha[t] * y_ * ht_pred[top_model])
            D /= np.sum(D)
            self.clfs.append(top_model)

            if (t+1) in cks:
                yield self



if __name__ == "__main__":
    
    MODEL_NAMES: List[str] = ["my_ada", "distboost", "preweak"]
    options: Dict[str, Any] = manage_options()
    print("Configuration:\n", json.dumps(vars(options), indent=4, sort_keys=True))
    assert options.model in MODEL_NAMES, "Model %s not supported!" %options.model
    
    MODEL: str = options.model
    DATASET: str = options.dataset
    TAGS: List[str] = options.tags.split(",")
    wandb.init(project='FederatedAdaboost',
               entity='mlgroup',
               name="%s_%s" %(MODEL, DATASET),
               tags=[DATASET, MODEL] + TAGS,
               config=options)
    
    TEST_SIZE: float = options.test_size
    NORMALIZE: bool = options.normalize
    N_CLIENTS: int = options.n_clients
    SEED: int = options.seed

    WEAK_LEARNER = DecisionTreeClassifier(random_state=SEED, max_depth=3)
    N_ESTIMATORS: List[int] = [1] + list(range(10, 300, 10))
    METRICS: List[str] = ["accuracy", "precision", "recall", "f1"]

    X_train, X_test, y_train, y_test = load_binary_classification_dataset(name_or_path=DATASET,
                                                                          test_size=TEST_SIZE,
                                                                          normalize=NORMALIZE,
                                                                          seed=SEED)
    print("# weak learners: %s" %N_ESTIMATORS)
    print("Training set size: %d" %X_train.shape[0])
    print("Test set size: %d" %X_test.shape[0])
    X_tr, y_tr = split_dataset(X_train, y_train, N_CLIENTS)

    if MODEL == "my_ada": 
        model = Adaboost(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_train, y_train
    elif MODEL == "distboost":
        model = Distboost(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    elif MODEL == "preweak":
        model = Preweak(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    else:
        raise ValueError("Unknown model %s." %MODEL)

    print("Training...")
    for strong_learner in model.fit(X_, y_, N_ESTIMATORS):
        y_pred_tr = strong_learner.predict(X_train)
        y_pred_te = strong_learner.predict(X_test)
        step = strong_learner.num_weak_learners()

        wandb.log({
            "train" : {
                "n_estimators" : step,
                "accuracy": accuracy_score(y_train, y_pred_tr), 
                "precision": precision_score(y_train, y_pred_tr),
                "recall": recall_score(y_train, y_pred_tr),
                "f1": f1_score(y_train, y_pred_tr)
            },
            "test" : {
                "n_estimators" : step,
                "accuracy": accuracy_score(y_test, y_pred_te), 
                "precision": precision_score(y_test, y_pred_te),
                "recall": recall_score(y_test, y_pred_te),
                "f1": f1_score(y_test, y_pred_te)
            }
        }, step=step)
    print("Training complete!")