import itertools
import rich
import typer

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

console = Console()
app = typer.Typer()

EXPS = "adult,letter,forestcover,splice,vehicle,vowel,segmentation,kr-vs-kp,sat,pendigits".split(",")
SEEDS = [0, 1, 2, 3]
MODELS = "samme,distsamme,preweaksamme,adaboost.f1".split(",")
NONIID = "uniform,num_examples_skw,lbl_skw,dirichlet_lbl_skw,pathological_skw,covariate_shift".split(",")

def experiment_to_skip(ds:str, seed:int, model:str, noniid:str, verbose:bool):
    """
    Returns True if the experiment should be skipped.
    Presently: datasets "adult" e "kr-vs-kp" should be skipped when iidness is in 
        "lbl_skw", "dirichlet_lbl_skw", "pathological_skw" (these are binary datasets which cannot
        work with these types of non-iidness)
    In addition: if noniid is "pathological_skw", then we are avoiding the experiments since there is
        a problem in how the data are split in some cases.
    """
    if ds in ["adult", "kr-vs-kp"] and noniid in ["lbl_skw", "dirichlet_lbl_skw", "pathological_skw"]:
        if verbose:
            console.log(f"[bold yellow]Skipping[/] {ds} {seed} {model} {noniid}")

        return True

    if noniid == "pathological_skw":
        if verbose:
            console.log(f"[bold yellow]Skipping[/] {ds} {seed} {model} {noniid}")
        return True

    return False

def experiment_list(verbose:bool=False):
    experiments = list(itertools.product(EXPS, SEEDS, MODELS, NONIID))
    result = []
    skipped = []

    for experiment in experiments:
        ds, seed, model, noniid = experiment

        if experiment_to_skip(ds, seed, model, noniid, verbose):
            skipped.append(experiment)
        else:        
            result.append(experiment)

    return result, skipped


@app.command()
def stats(verbose:bool=False):
    """
    Prints statistics about the experiments to be generated.
    """
    experiments, skipped = experiment_list(verbose)

    markdown = f"""
# Stats 
- experiments: {len(experiments)}
- skipped: {len(skipped)}
- **Total**: {len(experiments) + len(skipped)}
"""

    console.print(Panel.fit(Markdown(markdown), width=40))
        
@app.command()
def run(outfile:str=typer.Argument("Makefile"), verbose:bool=False, test_run:bool=True):
    """
    Generates a Makefile allowing to launch the experiments presented in  (Polato, Esposito, et al. 2022)
    """

    test_run_opt = "--test-run" if test_run else "--no-test-run"

    experiments, _ = experiment_list()

    with open(outfile, "w") as f:
        experiment_tags = []
        for experiment in experiments:
            ds, seed, model, noniid = experiment

            experiment_tags.append(f"logs/ijcnnexps_ds_{ds}_model_{model}_noniid_{noniid}_seed_{seed}.log")
            print(f"{experiment_tags[-1]}:", file=f)
            print(f"\tpython3 ijcnn_exps.py --seed={seed} --n-clients=10 --model={model} --non-iidness={noniid} --tags=IJCNN {test_run_opt} {ds}", file=f)

        exp_tags_str = " ".join(experiment_tags)
        print(f"all:{exp_tags_str}", file=f)

        print(f"clean_all_logs:", file=f)
        print(f"\trm -f logs/*.log", file=f)
        print(f"\trm -f logs/*.err", file=f)


if __name__ == '__main__':
    app()
