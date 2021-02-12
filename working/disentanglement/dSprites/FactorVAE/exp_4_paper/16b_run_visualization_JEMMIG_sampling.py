from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
run_id = "4b_FactorVAE_tc100"
# run_id = "3b_FactorVAE_tc50"

output_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "JEMMIG_sampling_plot"))
JEMMIG_sampling_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "JEMMIG"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,

    "JEMMIG_sampling_dir": JEMMIG_sampling_dir,
    "num_samples": 10000,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./16a_visualization_JEMMIG_sampling.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)