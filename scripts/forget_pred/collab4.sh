
#!/bin/bash

for ((seed = 5 ; seed < 10 ; seed++ ))
do
for method in baseline knn knnb svd
do

python collab_filter.py ${method} olmo-7b-inst-mb-nll-full-ood --thres -100.0 --known_k 30 --seed ${seed}



done
done

##!/bin/bash
#
#
#for method in baseline knn knnb svd
#do
##python collab_filter.py ${method} t5l-seed50-full-fpd-split --thres -100.0
##python collab_filter.py ${method} t5l-seed50-full-fpd-split --thres -100.0 --impute
#
#
#python collab_filter.py ${method} olmo-7b-inst-mb-nll --thres -100.0 --seed
##
##python collab_filter.py ${method} olmo-7b-dolma-nll-split-ss --thres -100.0
##
##python collab_filter.py ${method} olmo-7b-inst-mb-nll --thres -100.0 --impute
##
##python collab_filter.py ${method} olmo-7b-dolma-nll-split-ss --thres -100.0 --impute
#
#
#done