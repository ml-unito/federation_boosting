# Python command line: python ijcnn_exps.py --seed=0 --nclients=10 --model=adaboost.f1 --noniid=3 --labels=IJCNN pendigits

EXPS = "adult,letter,forestcover,splice,vehicle,vowel,segmentation,kr-vs-kp,sat,pendigits".split(",")
SEEDS = [0, 1, 2, 3]
MODELS = "samme,distsamme,preweaksamme,adaboost.f1".split(",")
NONIID= [0, 1, 2, 3, 4, 5]

experiment_tags = []
for exp in EXPS:
    for seed in SEEDS:
        for model in MODELS:
            for noniid in NONIID:
                experiment_tags.append("{}_{}_{}_{}".format(exp, seed, model, noniid))
                print("{}:".format(experiment_tags[-1]))
                print("\techo python ijcnn_exps.py --seed={} --nclients=10 --model={} --noniid={} --labels=IJCNN {}".format(seed, model, noniid, exp, exp))
                print("\tsleep 1")

print("all:{}".format(" ".join(experiment_tags)))

