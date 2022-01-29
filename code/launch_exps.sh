
datasets="breast sonar ionosphere banknote spambase particle forestcover12"
labels="EXP:1_OPT,WL:DT3"

for dataset in $datasets; do
    echo "Running experiment for $dataset"
    python3 fed_adaboost.py --model=my_ada --labels=$labels $dataset
    python3 fed_adaboost.py --model=preweak --labels=$labels $dataset
    python3 fed_adaboost.py --model=distboost --labels=$labels $dataset
done

