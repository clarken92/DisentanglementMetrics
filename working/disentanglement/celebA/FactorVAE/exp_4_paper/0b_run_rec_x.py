from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, RAW_DATA_DIR


run_id = "1b_like0b_z65"
output_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", run_id))
save_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary", "rec_x"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,
}

arg_str = get_arg_string(config)
set_GPUs([1])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./0a_rec_x.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)