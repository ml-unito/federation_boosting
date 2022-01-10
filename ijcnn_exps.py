# -*- coding: utf-8 -*-
from __future__ import division, print_function, annotations
from typing import Tuple, List, Generator, Optional
from copy import deepcopy
from math import log, ceil
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Annotated

import rich.traceback as traceback
from rich.console import Console

from optparse import OptionParser
import wandb
import noniid
import typer
from enum import Enum

# Handlers

app = typer.Typer()
console = Console(record=True)
traceback.install(show_locals=True)

from fed_adaboost import Boosting, split_dataset

# Enum type for samme, distsamme,preweaksamme and adaboost.f1
class FedAlgorithms(Enum):
    samme = "samme"
    distsamme = "distsamme"
    preweaksamme = "preweaksamme"
    adaboost = "adaboost.f1"

# Enum type for datasets: adult, letter, forestcover, splice, vehicle, vowel, segmentation, kr-vs-kp, sat and pendigits
class Datasets(Enum):
    adult = "adult"
    letter = "letter"
    forestcover = "forestcover"
    splice = "splice"
    vehicle = "vehicle"
    vowel = "vowel"
    segmentation = "segmentation"
    kr_vs_kp = "kr-vs-kp"
    sat = "sat"
    pendigits = "pendigits"

# Enum types for noniidness kinds
#    0 - uniform distribution (iid)
#    1 - examples' quantity skewness
#    2 - labels skewness
#    3 - Dirichlet distributed labels skewness
#    4 - pathological labels skewness
#    5 - covariate shift.
class Noniidness(Enum):
    uniform = "uniform"
    num_examples_skw = "num_examples_skw"
    lbl_skw = "lbl_skw"
    dirichlet_lbl_skw = "dirichlet_lbl_skw"
    pathological_skw = "pathological_skw"
    covariate_shift = "covariate_shift"

RANDOM_SEEDS = [98765, 12345, 999999, 101010, 765432, 171717, 13579, 24680]


def load_classification_dataset(name: str,
                                test_size: float=0.2, #real range (0,1)
                                seed: int=98765) -> Tuple[np.ndarray, np.ndarray]:
    if name == "letter":
        df = pd.read_csv("data/letter.csv", header=None)
        feats = ["feat_%d" %i for i in range(df.shape[1]-1)]
        df.columns = ["label"] + feats
        X = df[feats].to_numpy()
        y = LabelEncoder().fit_transform(df["label"].to_numpy())
    elif name == "pendigits":
        df_tr = pd.read_csv("data/pendigits.tr.csv", header=None)
        df_te = pd.read_csv("data/pendigits.te.csv", header=None)
        y_tr = df_tr.loc[:, 16].to_numpy()
        y_te = df_te.loc[:, 16].to_numpy()
        X_tr = df_tr.loc[:, :15].to_numpy()
        X_te = df_te.loc[:, :15].to_numpy()
        return X_tr, X_te, y_tr, y_te
    elif name == "sat":
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
    elif name in {"segmentation", "adult"}:
        X_tr, y_tr = load_svmlight_file("data/" + name + ".tr.svmlight")
        X_te, y_te = load_svmlight_file("data/" + name + ".te.svmlight")
        X_tr = X_tr.toarray()
        X_te = X_te.toarray()
        return X_tr, X_te, y_tr, y_te
    else:
        X, y = load_svmlight_file("data/" + name + ".svmlight")
        X = X.toarray()
        y = y.astype("int")
        y = LabelEncoder().fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=seed)


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
                y_pred[j, int(c)] += self.alpha[i]
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
                y_pred[j, int(c)] += 1
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
                yield self


class AdaboostF1(MulticlassBoosting):

    def federated_dist(self: AdaboostF1,
                       D: np.ndarray,
                       X: List[np.ndarray],
                       j: int) -> np.ndarray:
        min_index = sum([len(X[i]) for i in range(j)])
        D_ = D[min_index : min_index + X[j].shape[0]]
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
            best_error = errors[np.argmin(errors)]

            predictions = best_clf.predict(X_)

            self.alpha.append(log((1.0 - best_error) / (best_error + 1e-10)) + log(self.K-1)) # kind of additive smoothing
            D *= np.exp(self.alpha[t] * (y_ != predictions))

            D /= np.sum(D)
            self.clfs.append(best_clf)

            if (t+1) in cks:
                yield self


def distribute_dataset(X, y, n, non_iidness, seed):
    np.random.seed(seed)
    if non_iidness == Noniidness.uniform:
        return split_dataset(X, y, n)
    elif non_iidness == Noniidness.num_examples_skw:
        #ass = noniid.quantity_skew(X, y, n)
        ass = noniid.class_wise_quantity_skew(X, y, n)
    elif non_iidness == Noniidness.lbl_skw:
        nxc = max(2, ceil(len(set(y)) / n))
        ass = noniid.quantity_skew_lbl(X, y, n, class_per_client=nxc)
    elif non_iidness == Noniidness.dirichlet_lbl_skw:
        ass = noniid.dist_skew_lbl(X, y, n, beta=.5)
    elif non_iidness == Noniidness.pathological_skw:
        ass = noniid.pathological_dist_skew_lbl(X, y, n, shards_per_client=2)
    elif non_iidness == Noniidness.covariate_shift:
        ass = noniid.covariate_shift(X, y, n, modes=2)
    else:
        raise ValueError("Unknown non_iidness code %d!" %non_iidness)

    X_tr = [X[a] for a in ass]
    y_tr = [y[a] for a in ass]
    return X_tr, y_tr


@app.command()
def run(dataset: Datasets = typer.Argument(...), 
        seed: int = typer.Option(0, 
                                help="Pseudo-random seed for replicability purposes - default=98765. Seeds from"
                                     " 0 to 7 are automatically mapped to 'better' seeds."),
        test_size:float=typer.Option(0.2,
                                help="Test set size in percentage (0,1)"), 
        n_clients:int=typer.Option(10,
                                help="Number of clients (>= 1)"),
        model:FedAlgorithms=typer.Option("samme", 
                                 help="The model to train and test."),
        normalize:bool=typer.Option(False, help="Whether the instances has to be normalized or not"),                                
        non_iidness:Noniidness=typer.Option("uniform", help="Whether the instances have to be distributed in a non-iid way."),
        tags:str=typer.Option("", help="list of comma separated tags to be added in the wandb lod"),
        test_run:bool=typer.Option(True, help="Launch the script without WANDB support and training a single WL"),):
    """
    Testing Samme, Distboost and Preweak adaptation of Samme (Cooper et al. 2017) "
    as well as Adaboost.F1 (Polato, Esposito, et al. 2022) for multi-class classification. "
    """

    options = locals()
    
    MODEL: str = model.value
    DATASET: str = dataset.value
    TAGS: List[str] = [DATASET, MODEL, non_iidness.value]
    TAGS += tags.split(",") if "," in tags else []
    WANDB = not test_run
    options["tags"] = TAGS

    console.log("Configuration:", options, style="bold green")
    if WANDB:
        wandb.init(project='FederatedAdaboost',
                   entity='mlgroup',
                   name="%s_%s" %(MODEL, DATASET),
                   tags=TAGS,
                   config=options)
    
    TEST_SIZE: float = test_size
    NORMALIZE: bool = normalize
    N_CLIENTS: int = n_clients
    SEED: int = RANDOM_SEEDS[seed] if seed < 8 else seed
    NON_IIDNESS = non_iidness

    WEAK_LEARNER = DecisionTreeClassifier(random_state=SEED, max_leaf_nodes=10)
    N_ESTIMATORS: List[int] = [1] if test_run else list(range(10, 301, 10))

    X_train, X_test, y_train, y_test = load_classification_dataset(name=DATASET,
                                                                   test_size=TEST_SIZE,
                                                                   seed=SEED)
    if NORMALIZE:
        scaler = StandardScaler().fit(X_train)   
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    console.log(f"# weak learners: {N_ESTIMATORS}", style="italic")
    console.log(f"Training set size: {X_train.shape[0]}", style="italic")
    console.log(f"Test set size: {X_test.shape[0]}", style="italic")
    console.log(f"# classes: {len(set(y_train))}", style="italic")
    X_tr, y_tr = distribute_dataset(X_train, y_train, N_CLIENTS, NON_IIDNESS, SEED)

    if MODEL == "samme": 
        model = Samme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_train, y_train
    elif MODEL == "distsamme":
        model = DistSamme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    elif MODEL == "preweaksamme":
        model = PreweakSamme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    elif MODEL == "adaboost.f1":
        model = AdaboostF1(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    else:
        raise ValueError("Unknown model %s." %MODEL)

    console.log("Training...", style = "bold green")
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
                "f1": f1_score(y_train, y_pred_tr, average="micro")
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
        else: console.log(log_dict)

    console.log("Training complete!")
    console.save_text(
        f"logs/ijcnnexps_ds_{DATASET}_model_{MODEL}_noniid_{NON_IIDNESS.value}_seed_{seed}.log")

if __name__ == "__main__":
    typer.run(run)
