
datasets="breast sonar ionospehere banknote spambase particle forestcover12"

for dataset in $datasets; do
    echo "Running experiment for $dataset"
    python3 fed_adaboost.py --model=adaboost $dataset
    python3 fed_adaboost.py --model=my_ada $dataset
    python3 fed_adaboost.py --model=preweak $dataset
    python3 fed_adaboost.py --model=distboost $dataset
done

