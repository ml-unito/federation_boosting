from __future__ import division, print_function, annotations

from sklearn.base import ClassifierMixin
from typing import Tuple, List, Generator, Optional
from fed_adaboost import Boosting
from sklearn.tree import DecisionTreeClassifier
from math import log
from numpy.random import choice
from copy import deepcopy


import numpy as np


class MulticlassBoosting(Boosting):
    def __init__(self: MulticlassBoosting,
                 n_clf: int = 10,
                 clf_class: ClassifierMixin = DecisionTreeClassifier()):
        super(MulticlassBoosting, self).__init__(n_clf, clf_class)
        self.K = None  # to be defined in the fit method

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
            checkpoints: Optional[List[int]] = None,
            seed: int = 42,
            num_labels = None) -> Generator[Samme]:

        np.random.seed(seed)
        self.K = len(set(y)) if not num_labels else num_labels # assuming that all classes are in y
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
            # kind of additive smoothing
            self.alpha.append(
                log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1))
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
            checkpoints: Optional[List[int]] = None,
            seed: int = 42) -> Generator[DistSamme]:

        np.random.seed(seed)
        # assuming that all classes are in y
        self.K = len(set(np.concatenate(y)))
        cks = set(checkpoints) if checkpoints is not None else [self.n_clf]
        n_samples = sum([x.shape[0] for x in X])
        D = [np.full(x.shape[0], (1 / n_samples)) for x in X]
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            ht = []
            for j, X_ in enumerate(X):
                clf = deepcopy(self.clf_class)
                ids = choice(X_.shape[0], size=X_.shape[0],
                             replace=True, p=D[j]/np.sum(D[j]))
                X__, y__ = X_[ids], y[j][ids]
                clf.fit(X__, y__)
                ht.append(clf)

            H = Hyp(ht, self.K)
            self.clfs.append(H)

            min_error = 0
            predictions = []
            for j, X_ in enumerate(X):
                predictions.append(H.predict(X_))
                # / np.sum(D[j])
                min_error += np.sum(D[j][y[j] != predictions[j]])
            # kind of additive smoothing
            self.alpha.append(
                log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1))

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
            checkpoints: Optional[List[int]] = None,
            seed: int = 42) -> Generator[PreweakSamme]:

        np.random.seed(seed)
        # assuming that all classes are in y
        self.K = len(set(np.concatenate(y)))
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
        ht_pred = {h: h.predict(X_) for h in ht}
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
                err = np.sum(D[y__ != hpred[ids]])  # / np.sum(D)
                if err < min_error:
                    top_model = h
                    min_error = err

            # kind of additive smoothing
            self.alpha.append(
                log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1))
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
        D_ = D[min_index: min_index + X[j].shape[0]]
        return D_ / sum(D_)

    def fit(self: AdaboostF1,
            X: List[np.ndarray],
            y: List[np.ndarray],
            checkpoints: Optional[List[int]] = None,
            seed: int = 42) -> Generator[AdaboostF1]:

        np.random.seed(seed)
        # assuming that all classes are in y
        self.K = len(set(np.concatenate(y)))
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

            errors = np.array([sum(D[y_ != clf.predict(X_)])
                              for clf in fed_clfs])
            best_clf = fed_clfs[np.argmin(errors)]
            best_error = errors[np.argmin(errors)]

            predictions = best_clf.predict(X_)

            # kind of additive smoothing
            self.alpha.append(
                log((1.0 - best_error) / (best_error + 1e-10)) + log(self.K-1))
            D *= np.exp(self.alpha[t] * (y_ != predictions))

            D /= np.sum(D)
            self.clfs.append(best_clf)

            if (t+1) in cks:
                yield self
