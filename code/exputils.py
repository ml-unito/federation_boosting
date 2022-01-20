from typing import Tuple
from itertools import product

from rich.console import Console
import typer

console = Console()
app = typer.Typer()

ExpDescription = Tuple[str, int, str, str]

EXPS = "adult,letter,forestcover,splice,vehicle,vowel,segmentation,kr-vs-kp,sat,pendigits".split(
    ",")
SEEDS = [0, 1, 2, 3, 4]
MODELS = "samme,distsamme,preweaksamme,adaboost.f1".split(",")
NONIID = "uniform,num_examples_skw,lbl_skw,dirichlet_lbl_skw,pathological_skw,covariate_shift".split(
    ",")

def plot_ds_name(ds: str) -> str:
    return f"Datasets.{ds}"

def plot_iidness_name(iidness: str) -> str:
    return f"Noniidness.{iidness}"

def plot_fname(dataset: str, noniidness: str) -> str:
    if dataset == "Datasets.kr-vs-kp":
        dataset = "Datasets.kr_vs_kp"

    skw_maps = {"Noniidness.uniform": "uniform", 
                "Noniidness.num_examples_skw":"num_examples", 
                "Noniidness.lbl_skw":"lbl",
                "Noniidness.dirichlet_lbl_skw":"dirichlet", 
                "Noniidness.pathological_skw":"pathological", 
                "Noniidness.covariate_shift": "covariate_shift" }
    return f"images/skw/{skw_maps[noniidness]}/f1_{dataset}_{noniidness}.pdf"

def plotlist(verbose=False) -> list[str, str]:
    def plot_to_skip(exp):
        ds, iidness = exp
        return experiment_to_skip(ds, None, None, iidness, verbose)


    def to_plot_names(exp):
        ds, iidness = exp
        return plot_ds_name(ds), plot_iidness_name(iidness)

    filtered_list = list(filter(lambda exp: not plot_to_skip(exp),product(EXPS, NONIID)))

    return list(map(lambda exp: to_plot_names(exp), filtered_list))


def experiment_to_skip(ds: str, seed: int, model: str, noniid: str, verbose: bool):
    """
    Returns True if the experiment should be skipped.
    Presently: datasets "adult" e "kr-vs-kp" should be skipped when iidness is in 
        "lbl_skw", "dirichlet_lbl_skw", "pathological_skw" (these are binary datasets which cannot
        work with these types of non-iidness)        
    """
    if ds in ["adult", "kr-vs-kp"] and noniid in ["lbl_skw", "dirichlet_lbl_skw", "pathological_skw"]:
        if verbose:
            console.log(
                f"[bold yellow]Skipping[/] {ds} {seed} {model} {noniid}")

        return True

    # if noniid == "pathological_skw":
    #     if verbose:
    #         console.log(f"[bold yellow]Skipping[/] {ds} {seed} {model} {noniid}")
    #     return True

    return False


def experiment_list(verbose: bool = False) -> Tuple[list[ExpDescription], list[ExpDescription]]:
    """
    Returns a list of all the experiments that should be launched (i.e., the 
    list of all experiments minus the ones that should be skipped).

    if verbose is True, then skipped experiments are logged to stdout.
    """
    experiments: list[ExpDescription] = list(product(EXPS, SEEDS, MODELS, NONIID))
    result = []
    skipped = []

    for experiment in experiments:
        ds: str
        seed: int
        model: str
        noniid: str

        ds, seed, model, noniid = experiment

        if experiment_to_skip(ds, seed, model, noniid, verbose):
            skipped.append(experiment)
        else:
            result.append(experiment)

    return result, skipped

def hm(model):
    """
    Hilights model name if it contains "FedAlgorithms.adaboost"
    """
    if model == "FedAlgorithms.adaboost" or model == "FedAlgorithms.adaboost.f1":
        return "[bold]FedAlgorithms.adaboost.F1[/]"
    else:
        return model
