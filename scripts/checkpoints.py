import os
import sys

seed = sys.argv[1]

for ckstep in [1000, 10000, 20000, 30000, 40000, 50000]:
    os.system(f'python ocl.py --config_files configs/ocl.yaml configs/pt_snli/seed_ckpt.yaml --templates seed={seed} ckstep={ckstep}')