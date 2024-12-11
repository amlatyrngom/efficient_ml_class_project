#!/bin/bash

source /etc/profile
module load anaconda/Python-ML-2024b
./supercloud_data_load.sh
python sweep_results.py
