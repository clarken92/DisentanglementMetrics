#!/bin/bash

PYTHON_EXE="/home/dkdo/InstalledSoftwares/anaconda3/envs/PommerMan/bin/python"

BASE_PATH="/home/dkdo/Working/Workspace/Github/DeepLearningProjects"
MY_PROJECT="$BASE_PATH/GAN_VAE"
MY_UTILS="$BASE_PATH/MyUtils"

export PYTHONPATH="$MY_UTILS:$MY_PROJECT:$PYTHONPATH"

$PYTHON_EXE ./_run.py