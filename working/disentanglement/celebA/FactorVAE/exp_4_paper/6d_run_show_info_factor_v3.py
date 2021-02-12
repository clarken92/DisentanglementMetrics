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
# run_id = "5_tc50_zdim200"
run_id = "6_VAE_beta50"
# run_id = "6a_VAE_beta50"
# run_id = "6b_VAE_beta50"
# run_id = "8_VAE"

# enc_dec_model = "my"
# run_id = "0b_my_multiSave"

output_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary", "show_info_factors_v3"))
informativeness_metrics_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE",
                                           "auxiliary", "informativeness_metrics_v3"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,
    "informativeness_metrics_dir": informativeness_metrics_dir,

    # "num_bins": 100,
    # "bin_limits": "-4;4",
    # "data_proportion": 0.1,

    "num_bins": 100,
    "bin_limits": "-4;4",
    "data_proportion": 1.0,
    "top_k": 10,
    # "top_k": 45,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./6c_show_info_factors_v3.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)