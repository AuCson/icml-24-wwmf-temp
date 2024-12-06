#!/bin/bash

for ((seed = 0 ; seed < 5 ; seed++ ))
do
for method in baseline knn knnb svd
do

python collab_filter.py ${method} olmo-7b-dolma-nll-split-ss-full-ood \
 --thres -100.0 --known_k 30 --seed ${seed}


done
done