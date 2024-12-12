#!/bin/bash

sbatch --output="/home/gridsan/zzhang/projects/efficient_ml_class_project/slurm_logs/opt-125m-sweep.log" \
       --exclusive -N 1 batch_nanoquant.sh