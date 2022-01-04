import itertools


# Python command line: python ijcnn_exps.py --seed=0 --nclients=10 --model=adaboost.f1 --noniid=3 --labels=IJCNN pendigits

EXPS = "adult,letter,forestcover,splice,vehicle,vowel,segmentation,kr-vs-kp,sat,pendigits".split(",")
SEEDS = [0, 1, 2, 3]
MODELS = "samme,distsamme,preweaksamme,adaboost.f1".split(",")
NONIID= [0, 1, 2, 3, 4, 5]

experiments = list(itertools.product(EXPS, SEEDS, MODELS, NONIID))

experiment_tags = []
for experiment in experiments:
    ds, seed, model, noniid = experiment
    experiment_tags.append("ijcnnexps_ds_{}_model_{}_noniid_{}_seed_{}".format(ds, model, noniid, seed))
    print("{}:".format(experiment_tags[-1]))
    print("\techo python ijcnn_exps.py --seed={} --nclients=10 --model={} --noniid={} --labels=IJCNN {}".format(seed, model, noniid, ds))
    print("\tsleep 1")

print("all:{}".format(" ".join(experiment_tags)))

