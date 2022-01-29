import pandas as pd
import sys
from rich.console import Console
from itertools import product

console = Console()

data = pd.read_pickle(sys.argv[1]).groupby(['model', '_step'])

datasetname = sys.argv[1].split('.')[1].split('_')[0]
skwname = sys.argv[1].split('.')[2]

means = data.mean()
stds = data.std()

topf1 = means["test.f1"] + stds["test.f1"]
botf1 = means["test.f1"] - stds["test.f1"]

models = ["samme", "adaf1", "pw", "dist"]
metrics = ["f1", "botf1", "topf1"]

def ptos(p):
    m,c = p
    return m + "_" + c

cols = map(ptos, product(models, metrics))
result = pd.DataFrame(columns=["x", *cols ])

# print(list(data.model))


result.samme_f1 = means.loc['FedAlgorithms.samme','test.f1']
result.samme_topf1 = topf1['FedAlgorithms.samme']
result.samme_botf1 = botf1['FedAlgorithms.samme']

result.adaf1_f1 = means.loc['FedAlgorithms.adaboost', 'test.f1']
result.adaf1_topf1 = topf1['FedAlgorithms.adaboost']
result.adaf1_botf1 = botf1['FedAlgorithms.adaboost']

result.pw_f1 = means.loc['FedAlgorithms.preweaksamme', 'test.f1']
result.pw_topf1 = topf1['FedAlgorithms.preweaksamme']
result.pw_botf1 = botf1['FedAlgorithms.preweaksamme']

result.dist_f1 = means.loc['FedAlgorithms.distsamme', 'test.f1']
result.dist_topf1 = topf1['FedAlgorithms.distsamme']
result.dist_botf1 = botf1['FedAlgorithms.distsamme']
result.x = result.index

result.to_csv(f"tikz_csv/{datasetname}_{skwname}.csv", sep=' ', index=False)
