from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
run_id = "6_VAE_beta50"

output_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "independence_metrics"))
informativeness_metrics_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                                           "auxiliary", "informativeness_metrics_v3"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,
    "informativeness_metrics_dir": informativeness_metrics_dir,

    "num_bins": 100,
    "bin_limits": "-4;4",
    "data_proportion": 1.0,
    "top_k": -1,
}

arg_str = get_arg_string(config)
set_GPUs([1])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./9a_independence_metrics.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)