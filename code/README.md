[![python](https://img.shields.io/badge/PYTHON-blue?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)
[![version](https://img.shields.io/badge/python-3.8|3.9-blue?style=for-the-badge)]()
[![open-source](https://img.shields.io/badge/open%20source-blue?style=for-the-badge&logo=github&color=123456)](https://github.com/makgyver/)

# Programs and workflow

**`ijcnn_exps.py`**: performs a single experiment
**`expman`**: this experiment manager handle a number of tasks useful to organize and work with the experiments.
  - generates the makefile used to launch all the experiments
  - generates statistics about the experiments (how many are running, how many ended with errors, etc.)
  - generates the ranking tables

## Launching the experimentation

This is a two steps process:
  1. the Makefile needs to be generated using:
      ```
      ./expman generate-makefiles --no-test-run 
      ```
  2. the experiments can now be launched with:
      > make all

## Producing seaborn plots

Once the experiments completed seaborn plots can be generated using `mkgraphs.py`. 



<!-- # Federated Adaboost

Currently, the program tests Distboost and Preweak [[1]](#1) and compare their performance with a standard Adaboost (scikit-learn implementation + custom implementation).

## Requirements
Install the requirements through PyPi using the following command:
`pip install -r requirements.txt`

## Usage

```
Usage: ijcnn_exps.py [OPTIONS] DATASET:{adult|letter|forestcover|splice|vehicl
                     e|vowel|segmentation|kr-vs-kp|sat|pendigits}

Arguments:
  DATASET:{adult|letter|forestcover|splice|vehicle|vowel|segmentation|kr-vs-kp|sat|pendigits}
                                  [required]

Options:
  --seed INTEGER                  Pseudo-random seed for replicability
                                  purposes - default=98765. Seeds from 0 to 7
                                  are automatically mapped to 'better' seeds.
                                  [default: 0]
  --test-size FLOAT               Test set size in percentage (0,1)  [default:
                                  0.2]
  --n-clients INTEGER             Number of clients (>= 1)  [default: 10]
  --model [samme|distsamme|preweaksamme|adaboost.f1]
                                  The model to train and test.  [default:
                                  samme]
  --normalize / --no-normalize    Whether the instances has to be normalized
                                  or not  [default: no-normalize]
  --non-iidness [uniform|num_examples_skw|lbl_skw|dirichlet_lbl_skw|pathological_skw|covariate_shift]
                                  Whether the instances have to be distributed
                                  in a non-iid way.  [default: uniform]
  --tags TEXT                     list of comma separated tags to be added in
                                  the wandb lod
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```

## References
<a id="1">[1]</a>
J. Cooper and L. Reyzin, "Improved algorithms for distributed boosting," 2017 55th Annual Allerton Conference on Communication, Control, and Computing (Allerton), 2017, pp. 806-813, doi: 10.1109/ALLERTON.2017.8262822. -->
