import itertools
import rich
import typer

from rich.console import Console

console = Console()
app = typer.Typer()

EXPS = "adult,letter,forestcover,splice,vehicle,vowel,segmentation,kr-vs-kp,sat,pendigits".split(",")
SEEDS = [0, 1, 2, 3]
MODELS = "samme,distsamme,preweaksamme,adaboost.f1".split(",")
NONIID = "uniform,num_examples_skw,lbl_skw,dirichlet_lbl_skw,pathological_skw,covariate_shift".split(",")


@app.command()
def main(outfile:str=typer.Argument("Makefile")):
    """
    Generates a Makefile allowing to launch the experiments presented in  (Polato, Esposito, et al. 2022)
    """

    experiments = list(itertools.product(EXPS, SEEDS, MODELS, NONIID))

    with open(outfile, "w") as f:
        experiment_tags = []
        for experiment in experiments:
            ds, seed, model, noniid = experiment
            experiment_tags.append(
                "logs/ijcnnexps_ds_{}_model_{}_noniid_{}_seed_{}.log".format(ds, model, noniid, seed))
            print("{}:".format(experiment_tags[-1]), file=f)
            print("\tpython3 ijcnn_exps.py --seed={} --n-clients=10 --model={} --non-iidness={} --tags=IJCNN {}".format(seed, model, noniid, ds), file=f)


        print("all:{}".format(" ".join(experiment_tags)), file=f)


if __name__ == '__main__':
    typer.run(main)
