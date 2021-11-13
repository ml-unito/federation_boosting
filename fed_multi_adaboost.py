# -*- coding: utf-8 -*-
from __future__ import division, print_function, annotations
from os import name
from typing import Any, Tuple, Dict, List, Generator, Optional
from copy import deepcopy
from math import log
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from optparse import OptionParser
import json
import wandb

from fed_adaboost import Boosting, load_mnist_dataset, split_dataset, split_data_powerlaw

####################### ATTENTION #######################
# Set WANDB to True if you want to use Weights & biases #
#########################################################
WANDB = False                                           
#########################################################


def manage_options() -> Dict[str, Any]:
    parser = OptionParser(usage="usage: %prog [options] dataset",
                          version="%prog 0.1",
                          description="Testing Distboost and Preweak from Cooper et al. 2017 "\
                          "for multi-class classification. "\
                          "Supported dataset: iris, mnist, pandigits, letter and sat. The dataset "\
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
        print("Dataset argument missing. Please check the documentation: `python fed_multi_adaboost.py -h`")
        exit(0)
    options.dataset = args[0]
    return options


def load_classification_dataset(name_or_path: str,
                                test_size: float=0.1, #real range (0,1)
                                seed: int=42) -> Tuple[np.ndarray, np.ndarray]:
    #TODO: other datasets
    if name_or_path == "letter":
        df = pd.read_csv("data/letter.csv", header=None)
        feats = ["feat_%d" %i for i in range(df.shape[1]-1)]
        df.columns = ["label"] + feats
        X = df[feats].to_numpy()
        y = LabelEncoder().fit_transform(df["label"].to_numpy())
    elif name_or_path == "mnist":
        return load_mnist_dataset()
    elif name_or_path == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
    elif name_or_path == "nursery":
        df = pd.read_csv("data/nursery.csv", header=None)
        columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "label"]
        df.columns = columns
        y = LabelEncoder().fit_transform(df["label"].to_numpy())
        df = pd.concat([pd.get_dummies(df[c], prefix="oh_%s" %c) for c in columns[:-1]], axis=1)
        X = df.to_numpy(dtype=np.dtype(float))
        print(y)
    elif name_or_path == "pendigits":
        df_tr = pd.read_csv("data/pendigits.tr.csv", header=None)
        df_te = pd.read_csv("data/pendigits.te.csv", header=None)
        y_tr = df_tr.loc[:, 16].to_numpy()
        y_te = df_te.loc[:, 16].to_numpy()
        X_tr = df_tr.loc[:, :15].to_numpy()
        X_te = df_te.loc[:, :15].to_numpy()
        return X_tr, X_te, y_tr, y_te
    elif name_or_path == "sat":
        df_tr = pd.read_csv("data/sat.tr.csv", sep=" ", header=None)
        df_te = pd.read_csv("data/sat.te.csv", sep=" ", header=None)
        y_tr = df_tr.loc[:, 36].to_numpy()
        y_te = df_te.loc[:, 36].to_numpy()
        X_tr = df_tr.loc[:, :35].to_numpy()
        X_te = df_te.loc[:, :35].to_numpy()
        le = LabelEncoder().fit(y_tr)
        y_tr = le.transform(y_tr)
        y_te = le.transform(y_te)
        return X_tr, X_te, y_tr, y_te
    else:
        X, y = load_svmlight_file(name_or_path)
        y = LabelEncoder().fit_transform(y)
        X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    return X_train, X_test, y_train, y_test


class MulticlassBoosting(Boosting):
    def __init__(self: MulticlassBoosting,
                 n_clf: int=10,
                 clf_class: ClassifierMixin=DecisionTreeClassifier()):
        super(MulticlassBoosting, self).__init__(n_clf, clf_class)
        self.K = None #to be defined in the fit method

    def predict(self: Boosting,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0], self.K))
        for i, clf in enumerate(self.clfs):
            pred = clf.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, c] += self.alpha[i]
        return np.argmax(y_pred, axis=1)


class Samme(MulticlassBoosting):
    def fit(self: Samme,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]]=None,
            seed: int=42) -> Generator[Samme]:

        np.random.seed(seed)
        self.K = len(set(y)) # assuming that all classes are in y
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
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
            min_error = np.sum(D[y != predictions]) / np.sum(D)
            self.alpha.append(log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1)) # kind of additive smoothing
            D *= np.exp(self.alpha[t] * (y != predictions))
            D /= np.sum(D)
            self.clfs.append(clf)

            if (t+1) in cks:
                self.current_weak_error = min_error
                yield self


# Support class for Distboost
class Hyp():
    def __init__(self: Hyp,
                 ht: List[ClassifierMixin],
                 K: int):
        self.ht = ht
        self.K = K
    
    def predict(self: Hyp,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((X.shape[0], self.K))
        for h in self.ht:
            pred = h.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, c] += 1
        return np.argmax(y_pred, axis=1)


class DistSamme(MulticlassBoosting):
    def fit(self: DistSamme,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]]=None,
            seed: int=42) -> Generator[DistSamme]:

        np.random.seed(seed)
        self.K = len(set(np.concatenate(y))) # assuming that all classes are in y
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
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

            H = Hyp(ht, self.K)
            self.clfs.append(H)
            
            min_error = 0
            predictions = []
            for j, X_ in enumerate(X):
                predictions.append(H.predict(X_))
                min_error += np.sum(D[j][y[j] != predictions[j]]) #/ np.sum(D[j])
            self.alpha.append(log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1)) # kind of additive smoothing

            for j, X_ in enumerate(X):
                D[j] *= np.exp(self.alpha[t] * (y[j] != predictions[j]))
            Dsum = sum([np.sum(d) for d in D])
            
            for d in D:
                d /= Dsum


            if (t+1) in cks:
                self.current_weak_error = min_error
                yield self


class PreweakSamme(MulticlassBoosting):
    def fit(self: PreweakSamme,
            X: np.ndarray,
            y: np.ndarray,
            checkpoints: Optional[List[int]]=None,
            seed: int=42) -> Generator[PreweakSamme]:
        
        np.random.seed(seed)
        self.K = len(set(np.concatenate(y))) # assuming that all classes are in y
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
        ht = []
        for j, X_ in enumerate(X):
            clf = Samme(self.n_clf, self.clf_class)
            for h in clf.fit(X_, y[j], checkpoints):
                pass
            ht.extend(clf.clfs)

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
                err = np.sum(D[y__ != hpred[ids]]) #/ np.sum(D)
                if err < min_error:
                    top_model = h
                    min_error = err

            self.alpha.append(log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1)) # kind of additive smoothing
            D *= np.exp(self.alpha[t] * (y_ != ht_pred[top_model]))
            D /= np.sum(D)
            self.clfs.append(top_model)


            if (t+1) in cks:
                self.current_weak_error = min_error
                yield self


class AdaboostF1(MulticlassBoosting):

    def federated_dist(self:AdaboostF1, D:np.ndarray, X:List[np.ndarray], j:int):
        min_index = sum([len(X[i]) for i in range(j)])
        D_ = D[min_index:min_index+X[j].shape[0]]
        return D_ / sum(D_)


    def fit(self: AdaboostF1,
            X: List[np.ndarray],
            y: List[np.ndarray],
            checkpoints: Optional[List[int]]=None,
            seed: int=42) -> Generator[AdaboostF1]:

        np.random.seed(seed)
        self.K = len(set(np.concatenate(y))) # assuming that all classes are in y
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]

        # merge the datasets into one (not possible in a real distributed/federated scenario)
        X_ = np.vstack(X) 
        y_ = np.concatenate(y)

        n_samples = X_.shape[0]
        D = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        self.alpha = []

        for t in range(self.n_clf):
            fed_clfs = []
            for j, X__ in enumerate(X):
                D_ = self.federated_dist(D, X, j)
                n_samples_ = X__.shape[0]

                clf = deepcopy(self.clf_class)
                ids = choice(n_samples_, size=n_samples_, replace=True, p=D_)
                clf.fit(X__[ids], y[j][ids])
                fed_clfs.append(clf)


            errors = np.array([sum(D[y_ != clf.predict(X_)]) for clf in fed_clfs])
            best_clf = fed_clfs[np.argmin(errors)]
            min_error = errors[np.argmin(errors)]

            predictions = best_clf.predict(X_)

            self.alpha.append(log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1)) # kind of additive smoothing
            D *= np.exp(self.alpha[t] * (y_ != predictions))

            D /= np.sum(D)
            self.clfs.append(best_clf)


            if (t+1) in cks:
                self.current_weak_error = min_error
                yield self




if __name__ == "__main__":
    
    MODEL_NAMES: List[str] = ["samme", "distsamme", "preweaksamme", "adaboostf1"]
    options: Dict[str, Any] = manage_options()
    print("Configuration:\n", json.dumps(vars(options), indent=4, sort_keys=True))
    assert options.model in MODEL_NAMES, "Model %s not supported!" %options.model
    
    MODEL: str = options.model
    DATASET: str = options.dataset
    TAGS: List[str] = options.tags.split(",")
    if WANDB:
        wandb.init(project='FederatedAdaboost',
                   entity='mlgroup',
                   name="%s_%s" %(MODEL, DATASET),
                   tags=[DATASET, MODEL] + TAGS,
                   config=options)
    
    TEST_SIZE: float = options.test_size
    NORMALIZE: bool = options.normalize
    N_CLIENTS: int = options.n_clients
    SEED: int = options.seed

    WEAK_LEARNER = DecisionTreeClassifier(random_state=SEED, max_leaf_nodes=10)
    N_ESTIMATORS: List[int] = [1] + list(range(10, 300, 10))
    METRICS: List[str] = ["accuracy", "precision", "recall", "f1"]

    X_train, X_test, y_train, y_test = load_classification_dataset(name_or_path=DATASET,
                                                                   test_size=TEST_SIZE,
                                                                   seed=SEED)
    if NORMALIZE:
        scaler = StandardScaler().fit(X_train)   
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    print("# weak learners: %s" %N_ESTIMATORS)
    print("Training set size: %d" %X_train.shape[0])
    print("Test set size: %d" %X_test.shape[0])
    X_tr, y_tr = split_dataset(X_train, y_train, N_CLIENTS)
    X_, y_ = X_tr, y_tr

    if MODEL == "samme": 
        model = Samme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_train, y_train
    elif MODEL == "distsamme":
        model = DistSamme(max(N_ESTIMATORS), WEAK_LEARNER)
    elif MODEL == "preweaksamme":
        model = PreweakSamme(max(N_ESTIMATORS), WEAK_LEARNER)
    elif MODEL == "adaboostf1":
        model = AdaboostF1(max(N_ESTIMATORS), WEAK_LEARNER)
    else:
        raise ValueError("Unknown model %s." %MODEL)

    print("Training...")
    for strong_learner in model.fit(X_, y_, N_ESTIMATORS):
        y_pred_tr = strong_learner.predict(X_train)
        y_pred_te = strong_learner.predict(X_test)
        step = strong_learner.num_weak_learners()

        log_dict = {
            "train" : {
                "n_estimators" : step,
                "accuracy": accuracy_score(y_train, y_pred_tr), 
                "precision": precision_score(y_train, y_pred_tr, average="micro"),
                "recall": recall_score(y_train, y_pred_tr, average="micro"),
                "f1": f1_score(y_train, y_pred_tr, average="micro"),
                "weak_error": strong_learner.current_weak_error
            },
            "test" : {
                "n_estimators" : step,
                "accuracy": accuracy_score(y_test, y_pred_te), 
                "precision": precision_score(y_test, y_pred_te, average="micro"),
                "recall": recall_score(y_test, y_pred_te, average="micro"),
                "f1": f1_score(y_test, y_pred_te, average="micro")
            }
        }

        if WANDB: wandb.log(log_dict, step=step)
        else: print(log_dict)

    print("Training complete!")