import itertools

EXPS = "adult,letter,forestcover,splice,vehicle,vowel,segmentation,kr-vs-kp,sat,pendigits".split(",")
SEEDS = [0, 1, 2, 3]
MODELS = "samme,distsamme,preweaksamme,adaboost.f1".split(",")
NONIID = [0, 1, 2, 3, 4, 5]

# Generates a Makefile for the launching the experiments.
# Usage:
#   mk_makefile.py > Makefile

if __name__ == '__main__':
    experiments = list(itertools.product(EXPS, SEEDS, MODELS, NONIID))

    experiment_tags = []
    for experiment in experiments:
        ds, seed, model, noniid = experiment
        experiment_tags.append("ijcnnexps_ds_{}_model_{}_noniid_{}_seed_{}".format(ds, model, noniid, seed))
        print("{}:".format(experiment_tags[-1]))
        print("\t@echo python ijcnn_exps.py --seed={} --nclients=10 --model={} --noniid={} --labels=IJCNN {}".format(seed, model, noniid, ds))
        print("\t@sleep 1")

    print("all:{}".format(" ".join(experiment_tags)))

