from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
# run_id = "0_Gz10"
run_id = "0_Gz50"
# run_id = "0b_Gz50"
# run_id = "0c_Gz50"
# run_id = "1_Gz100"
# run_id = "2_Gz200"


output_dir = abspath(join(RESULTS_DIR, "celebA", "AAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "celebA", "AAE", "auxiliary", "show_info_factors_v3"))
informativeness_metrics_dir = abspath(join(RESULTS_DIR, "celebA", "AAE",
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
    "top_k": 45,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./6c_show_info_factors_v3.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)