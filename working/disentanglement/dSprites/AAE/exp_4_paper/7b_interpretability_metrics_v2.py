from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
run_ids = [
    "1_Gz50_zdim10",
    "1a_Gz50_zdim10",
    "1b_Gz50_zdim10",
    "1c_Gz50_zdim10",

    "2_Gz20_zdim10",
    "2a_Gz20_zdim10",
    "2b_Gz20_zdim10",
    "2c_Gz20_zdim10",

    "3_Gz10_zdim10",
    "3a_Gz10_zdim10",
    "3b_Gz10_zdim10",
    "3c_Gz10_zdim10",
]

save_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", "auxiliary", "interpretability_metrics_v2"))

for run_id in run_ids:
    # output_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", enc_dec_model, run_id))
    output_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", run_id))

    config = {
        "output_dir": output_dir,
        "save_dir": save_dir,

        # "num_bins": 100,
        "num_bins": 200,
        "bin_limits": "-4;4",
        "data_proportion": 1.0,
    }

    arg_str = get_arg_string(config)
    set_GPUs([1])

    print("Running arguments: [{}]".format(arg_str))
    run_command = "{} ./7a_interpretability_metrics_v2.py {}".format(PYTHON_EXE, arg_str).strip()
    subprocess.call(run_command, shell=True)