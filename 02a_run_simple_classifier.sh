#!/bin/bash
all_runs='RandomForest/n10@{"n_estimators": 10};n100@{"n_estimators": 100};n1000@{"n_estimators": 1000}
NaiveBayes/default@{}'

IFS=$'\n'

for run in $all_runs; do
    IFS='/' read model args <<< $run
    echo $model
    IFS=';'
    for arg in $args; do
    	echo "Running $model with $arg"
    	IFS="@" read cname conf <<< $arg
    	./02a_simple_classifier.py --data output/0-munich-data/som-10 --output output/simple-classifiers/$model-$cname --model_name $model --modelargs $conf
    done
    IFS=$'\n'
done
