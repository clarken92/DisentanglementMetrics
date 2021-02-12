from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
run_id = "0_Gz50"

output_dir = abspath(join(RESULTS_DIR, "celebA", "AAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "celebA", "AAE", "auxiliary", "z_corr_mat"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,
}

arg_str = get_arg_string(config)
set_GPUs([1])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./2a_z_corr_mat.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)