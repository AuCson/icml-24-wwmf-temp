# ICML-What-Will-My-Model-Forget

This repository contains uncleaned code for the paper "What Will My Model Forget?
Forecasting Forgotten Examples in Language Model Refinement".

Important entry files

- `instance_ocl_p3.py`: model refinement to fix prediction errors & perform statistics of forgetting
- `train_logit_predict.py`: training & evaluation of rep-based and logit-based forgetting prediction approaches
- `stat_logits.py`: obtaining logit statistics for logit-based forgetting prediction 

Dataset

- P3 dataset: We used preprocessed data from https://github.com/INK-USC/ReCross/tree/main/data
- FLAN: We used this version: https://huggingface.co/datasets/Muennighoff/flan

Side note

- `configs/p3/instance-bart0-base-ocl/lr1e-6.yaml` actually uses LR=1e-5 as reported in the paper.
- Fixed a sgn issue I messed up - now the implementation is faithful to icml paper instead of the follow-up paper.

