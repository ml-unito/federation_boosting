# -*- coding: utf-8 -*-
from __future__ import division, print_function, annotations
from typing import Tuple, List
from copy import deepcopy
from math import ceil
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from fed_adaboost import split_dataset
from pathlib import Path


import rich.traceback as traceback
from rich.console import Console

import noniid
from fed_algorithms import *

import wandb
import typer
from enum import Enum

# Handlers

app = typer.Typer()
console = Console(record=True)
error_console = Console(stderr=True)
traceback.install(show_locals=False)

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


def load_classification_dataset(name: str,
                                test_size: float = 0.2,  # real range (0,1)
                                seed: int = 98765) -> Tuple[np.ndarray, np.ndarray]:
    if name == "letter":
        df = pd.read_csv("data/letter.csv", header=None)
        feats = ["feat_%d" % i for i in range(df.shape[1]-1)]
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
    elif name == "forestcover":
        if not os.path.isfile("covtype.data"):
            dload.save_unzip(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz")
        covtype_df = pd.read_csv("covtype.data", header=None)
        covtype_df = covtype_df[covtype_df[54] < 3]
        X = covtype_df.loc[:, :53].to_numpy()
        y = (covtype_df.loc[:, 54] - 1).to_numpy()
        ids = permutation(X.shape[0])
        X, y = X[ids], y[ids]
        X_train, X_test = X[:250000], X[250000:]
        y_train, y_test = y[:250000], y[250000:]
        return X_train, X_test, y_train, y_test
    else:
        X, y = load_svmlight_file("data/" + name + ".svmlight")
        X = X.toarray()
        y = y.astype("int")
        y = LabelEncoder().fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=seed)


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
        ass = noniid.pathological_dist_skew_lbl(X, y, n, shards_per_client=3)
    elif non_iidness == Noniidness.covariate_shift:
        ass = noniid.covariate_shift(X, y, n, modes=2)
    else:
        raise ValueError("Unknown non_iidness code %d!" %non_iidness)

    X_tr = [X[a] for a in ass]
    y_tr = [y[a] for a in ass]
    return X_tr, y_tr

def get_model(model, X_train, y_train, X_tr, y_tr, N_ESTIMATORS, WEAK_LEARNER):
    if model.value == "samme":
        learner = Samme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_train, y_train
    elif model.value == "distsamme":
        learner = DistSamme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    elif model.value == "preweaksamme":
        learner = PreweakSamme(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    elif model.value == "adaboost.f1":
        learner = AdaboostF1(max(N_ESTIMATORS), WEAK_LEARNER)
        X_, y_ = X_tr, y_tr
    else:
        raise ValueError("Unknown model %s." % model.value)

    return learner, X_, y_

def scale_data(normalize, X_train, X_test):
    if normalize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test

def execute_experiment(dataset, seed, test_size, n_clients, model, normalize, non_iidness, tags, test_run):
    options = deepcopy(locals())

    WEAK_LEARNER = DecisionTreeClassifier(random_state=seed, max_leaf_nodes=10)
    N_ESTIMATORS: List[int] = [1] if test_run else [1] + list(range(10, 301, 10))
    TAGS = tags.strip().split(",") if tags else []
    options["tags"] = TAGS

    console.log("Configuration:", options, style="bold green")

    if not test_run:
        wandb.init(project='FederatedAdaboost',
                    entity='mlgroup',
                    name=f"{dataset.value}_{model.value}_{non_iidness.value}_{seed}",
                    tags=TAGS,
                    config=options)

    X_train, X_test, y_train, y_test = load_classification_dataset(name=dataset.value,
                                                                    test_size=test_size,
                                                                    seed=seed)

    X_train, X_test = scale_data(normalize, X_train, X_test)

    console.log(f"# weak learners: {N_ESTIMATORS}", style="italic")
    console.log(f"Training set size: {X_train.shape[0]}", style="italic")
    console.log(f"Test set size: {X_test.shape[0]}", style="italic")
    console.log(f"# classes: {len(set(y_train))}", style="italic")

    X_tr, y_tr = distribute_dataset(
        X_train, y_train, n_clients, non_iidness, seed)

    learner, X_, y_ = get_model(model, X_train, y_train, X_tr, y_tr, N_ESTIMATORS, WEAK_LEARNER)

    console.log("Training...", style="bold green")
    for strong_learner in learner.fit(X_, y_, N_ESTIMATORS):
        y_pred_tr = strong_learner.predict(X_train)
        y_pred_te = strong_learner.predict(X_test)
        step = strong_learner.num_weak_learners()

        log_dict = {
            "train": {
                "n_estimators": step,
                "accuracy": accuracy_score(y_train, y_pred_tr),
                "precision": precision_score(y_train, y_pred_tr, average="micro"),
                "recall": recall_score(y_train, y_pred_tr, average="micro"),
                "f1": f1_score(y_train, y_pred_tr, average="micro")
            },
            "test": {
                "n_estimators": step,
                "accuracy": accuracy_score(y_test, y_pred_te),
                "precision": precision_score(y_test, y_pred_te, average="micro"),
                "recall": recall_score(y_test, y_pred_te, average="micro"),
                "f1": f1_score(y_test, y_pred_te, average="micro")
            }
        }

        if not test_run:
            wandb.log(log_dict, step=step)
        else:
            console.log(log_dict)


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

    filename = f"logs/ijcnnexps_ds_{dataset.value}_model_{model.value}_noniid_{non_iidness.value}_seed_{seed}"
    run_file = filename+".run"
    err_file = filename+".err"
    log_file = filename+".log"

    Path(run_file).touch()

    try:
        execute_experiment(dataset, seed, test_size, n_clients, model, normalize, non_iidness, tags, test_run)

        console.log("Training complete!")
        console.save_text(log_file)
    except:
        console.print_exception(show_locals=True)
        console.save_text(err_file)

    Path(run_file).unlink()

if __name__ == "__main__":
    typer.run(run)
