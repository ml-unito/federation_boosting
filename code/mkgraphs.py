import pandas as pd
import wandb

from itertools import groupby, product
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

import typer
from rich.console import Console
from rich.table import Table
import rich.traceback as traceback
from rich import progress
from exputils import plotlist, plot_fname

console = Console()
traceback.install()
app = typer.Typer()

def avg_stats(runs:list, stat:str) -> pd.DataFrame:
    """
    Takes a list of repetitions for a given experiment and returns a numpy array with the averages
    of the history[stat] results.
    """
    stats = []
    for run in runs:
        run_stats = run.history()[stat]
        stats.append(run_stats.to_numpy())

    return np.mean(stats, axis=0), np.std(stats, axis=0)



@app.command()
def plot(dataset:str, non_iidness:str):
    """
    Plot the results of the runs
    """
    api:wandb.Api = wandb.Api()
    entity:str
    project:str 

    entity, project = "mlgroup", "FederatedAdaboost "
    runs = api.runs(entity + "/" + project, filters={"config.dataset": dataset, "config.non_iidness": non_iidness}, order="config.model")

    exps_by_model = groupby(runs, lambda r: r.config["model"])

    data = pd.DataFrame()
    for model,runs in exps_by_model:
        for run in runs:
            history = run.history()
            history = history.assign(model=model)
            data = data.append(history, ignore_index=True)

    try:
        fig,ax = plt.subplots(1,1)
        sns.lineplot(data=data, x="_step", hue="model", y="test.f1", ax=ax)
        plt.savefig(plot_fname(dataset, non_iidness))
        plt.close(fig)
    except(ValueError):
        console.print(f"Data error for {dataset} {non_iidness}")

@app.command()
def plot_all(verbose:bool=False):
    pl = plotlist(verbose=verbose)
    
    for plotelems in progress.track(pl, description="Plotting..."):
        plot(*plotelems)


if __name__ == "__main__":
    app()


