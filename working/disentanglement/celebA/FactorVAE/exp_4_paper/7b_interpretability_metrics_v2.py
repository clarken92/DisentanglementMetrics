from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
# run_id = "0a_1Konny_multiSave"
# run_id = "0b_1Konny_multiSave"
# run_id = "1_tc50_multiSave"
# run_id = "1a_tc50_multiSave"
# run_id = "1b_tc50_multiSave"
# run_id = "4_tc50_zdim100"
run_id = "5_tc50_zdim200"
# run_id = "6_VAE_beta50"
# run_id = "6a_VAE_beta50"
# run_id = "6b_VAE_beta50"
# run_id = "8_VAE"
# run_id = "9_VAE_zdim100"
# run_id = "10_VAE_zdim200"

# enc_dec_model = "my"

output_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary", "interpretability_metrics_v2"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,

    # "num_bins": 100,
    # "num_bins": 20,
    "num_bins": 50,

    "bin_limits": "-4;4",
    "data_proportion": 1.0,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./7a_interpretability_metrics_v2.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)