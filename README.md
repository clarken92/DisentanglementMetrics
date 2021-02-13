# Theory and Evaluation Metrics for Learning Disentangled Representations


This repository contains the official implementation of our paper:
> [**Theory and Evaluation Metrics for Learning Disentangled Representations**](https://arxiv.org/abs/1908.09961)
>
> [Kien Do](https://twitter.com/kien_do_92), [Truyen Tran](https://twitter.com/truyenoz)

__Accepted at [ICLR 2020](https://openreview.net/forum?id=HJgK0h4Ywr).__


## Contents
1. [Requirements](#requirements)
1. [Features](#features)
0. [Repository structure](#repository-structure)
0. [Setup](#setup)
0. [Training](#training)
0. [Testing](#testing)
0. [Reproducing results in our paper](#reproducing-results-in-our-paper)
0. [Citation](#citation)

## Requirements
Tensorflow >= 1.8

The code hasn't been tested with Tensorflow 2.

This repository is designed to be self-contained. If during running the code, some packages are required, these packages can be downloaded via pip or conda.
Please email me if you find any problems related to this.

## Features
- Support model saving
- Support logging
- Support tensorboard visualization

## Repository structure
Our code is organized in 3 main parts:
- `models`: Containing models used in our paper, which are AAE, FactorVAE and BetaVAE (a special case of FactorVAE).
- `utils`, `my_utils`: Containing utility functions.
- `working`: Containing scripts for training/testing models and reproducing results shown in our paper.

In our code, we define main models and their components (encoder, decoder, discriminator) separately. This allows us to use AAE, FactorVAE with different architectures of the encoder, decoder, and discriminator. The code for components is placed in the `models/enc_dec` folder.

## Setup
The setup for training is **very simple**. All you need to do is opening the `global_settings.py` file and changing the values of the global variables to match yours. The meanings of the global variables are given below:
* `PYTHON_EXE`: Path to your python interpreter.
* `PROJECT_NAME`: Name of the project, which I set to be `'DisentanglementMetrics'`.
* `PROJECT_DIR`: Path to the root folder containing the code of this project.
* `RESULTS_DIR`: Path to the root folder that will be used to store results for this project.  
* `RAW_DATA_DIR`: Path to the root folder that contains raw datasets. By default, the root directory of the CelebA dataest is `$RAW_DATA_DIR/ComputerVision/CelebA` and the root directory of the dSprites dataset is `$RAW_DATA_DIR/ComputerVision/dSprites`.

**IMPORTANT NOTE**: Since this repository is organized as a Python project, I strongly encourage you to import it as a project to an IDE (e.g., PyCharm). By doing so, the path to the root folder of this project will be automatically added to PYTHONPATH when you run the code via your IDE. Otherwise, you have to explicitly add it when you run in terminal. Please check `_run.sh` to see how it works.

## Training
Once you have setup everything in `global_settings.py`, you can start training by running the following command in your terminal:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python _train.py [required arguments]
```
**IMPORTANT NOTE**: If you run using the command above, please remember to provide all **required** arguments specified in `_train.py` otherwise errors will be raised.

However, if you are too lazy to type arguments in the terminal (like me :sweat_smile:), you can set these arguments in the `config` dictionary in `_run.py` and simply run the `_run.py` file:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python _run.py
```

I also provide a `_run.sh` file as an example for you.

## Testing
After training, you can test your models by running the following command:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python _test.py [required arguments]
```

Or you can set all arguments in `_run_test.py` and run:
 
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python _run_test.py
```

The test code will do the following:
* Show generated images from the prior distribution of latent variables
* Show reconstructed images
* Interpolate between two input images
* Show the correlation matrix and histogram of all latent variables

## Reproducing results in our paper
All scripts for reproducing results in our paper are placed in the folder `exp_4_paper`. Please check the file names for which experiments you want to do. 

## Citation
If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{do2019theory,
  title={Theory and evaluation metrics for learning disentangled representations},
  author={Do, Kien and Tran, Truyen},
  journal={arXiv preprint arXiv:1908.09961},
  year={2019}
}
```
