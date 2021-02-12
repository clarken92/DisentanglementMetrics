#!/bin/bash

USER="dkie"
PYTHON_EXE="/home/$USER/InstalledSoftwares/anaconda3/envs/Tensorflow/bin/python"
MY_PROJECT="/home/$USER/Working/Workspace/Github/DisentanglementMetrics"

export PYTHONPATH="$MY_PROJECT:$PYTHONPATH"
$PYTHON_EXE ./15b_run_SEPIN_sampling.py