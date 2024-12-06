import os

for seed in [1,2,3]:
    for ckstep in [50000]:
        os.system(f'python ocl.py --config_files configs/ocl.yaml configs/pt_snli/seed_ckpt.yaml --templates seed={seed} ckstep={ckstep} --max_ocl 100000 --max_pt 10000 --log_postfix _v3')