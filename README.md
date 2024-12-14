# NanoQuant: Investigating W4A4 Quantization

## Introduction
This repository contains the code and artifacts of the final project for [6.5940 TinyML](efficientml.ai) at MIT. The project aims to investigate the promises and limitations of W4A4 quantization for large language models (LLMs). The approach we take is to combine Activation-aware Weight Quantization ([AWQ](https://github.com/mit-han-lab/llm-awq)) and [SmoothQuant](https://github.com/mit-han-lab/smoothquant).
### Code Organization
Submodules `llm-awq` and `smoothquant` respectively contain the code for AWQ and SmoothQuant developed by their original authors as well as modifications we applied to make these approaches compatible together. `nanoquant.investigate` contains the main functions of our package. `nanoquant` and the root directory of the repository both have Jupyter notebooks containing the script and result for different base models. The contexts for the notebooks should be self-evident from the notebooks. The root directory contains various scripts, which we explain in more detail below.

## Setup
### Basic
To sync the submodules and install necessary dependencies, run
```sh
bash init_setup.sh
```
Especially if you have a previous installation of either AWQ or SmoothQuant from PyPI, you will need to `cd` into `llm-awq` and `smoothquant` to run the following to install the modified version for this projec.
```sh
pip install .
```

If you are on a cluster where the compute node does not have internet access, refer to `data_load.sh` to pre-load the data on the internet-enabled nodes.


### SmoothQuant 
To make vanilla SmoothQuant scales, use
```sh
bash make_smooth_scales.sh <short-model-name>
```
an example of `short-model-name` is `opt-125m`.
To make AWQ-aware SmoothQuant scales, use
```sh
bash make_smooth_awq_scales.sh <short-model-name>
```
These scripts contain some utility code to work on a cluster where the compute nodes do not have internet access. To run these script on supercloud, use `make_smooth_scales_supercloud.sh` and `make_smooth_awq_scales_supercloud.sh`. You may need to edit them to set appropriate local model paths.

This will store the corresponding SmoothQuant scales at `$smoothquant_path/act_scales/` (`smoothquant/act_scales` by default), which the rest of the package will rely on.

## Running NanoQuant
To run nanoquant, you can refer to `sweep_results.py` or any Jupyter Notebook in this repository. `sweep_results.py` manually loads a model from local disk. Change that utility code to other huggingface access methods as necessary.