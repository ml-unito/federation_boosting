[![python](https://img.shields.io/badge/PYTHON-blue?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)
[![version](https://img.shields.io/badge/python-3.8|3.9-blue?style=for-the-badge)]()
[![open-source](https://img.shields.io/badge/open%20source-blue?style=for-the-badge&logo=github&color=123456)](https://github.com/makgyver/)

# Federated Adaboost

Currently, the program tests Distboost and Preweak [[1]](#1) and compare their performance with a standard Adaboost (scikit-learn implementation + custom implementation).

## Requirements
Install the requirements through PyPi using the following command:
`pip install -r requirements.txt`

## Usage
`python fed_adaboost.py [options] dataset`

Options:

* `--version` \
    Show program's version number and exit
* `-h`, `--help` \
    Show "this" help message and exit
* `-s SEED`, `--seed=SEED` \
    Pseudo-random seed for replicability purposes - default=42
* `-t TEST_SIZE`, `--test_size=TEST_SIZE` \
    Test set size in percentage (0,1) - default=0.1
* `-n N_CLIENTS`, `--nclients=N_CLIENTS` \
    Number of clients (>= 1) - default=10
* `-z`, `--normalize`\
    Whether the instances has to be normalized (standard scaling) or not - default=False
* `-o`, `--output`\
    Whether the results has to be saved on a JSON file - default=False


**Supported datasets**: breast, mnistXY, sonar, ionospehere, banknote, spambase, particle, and forestcover12. Note that mnistXY is a binary classification subset of MNIST in which X is a digit and Y is another digit with X,Y in {0, 1, ..., 9}. The `test_size` argument is ignored for the mnist, particle and the forestcover12
datasets for which it is used a default split size. The dataset can be also a path to a svmlight file.

## References
<a id="1">[1]</a>
J. Cooper and L. Reyzin, "Improved algorithms for distributed boosting," 2017 55th Annual Allerton Conference on Communication, Control, and Computing (Allerton), 2017, pp. 806-813, doi: 10.1109/ALLERTON.2017.8262822.
