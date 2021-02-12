from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


# Tensorflow version < 1.13 use cuda 9.0
# We should use jax with cuda 9.0
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


for num_samples in [10000]: # [10000, 20000, 50000, 5000, 2000, 1000]:
    for run_id in run_ids:
        # output_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", enc_dec_model, run_id))
        output_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", run_id))
        save_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", "auxiliary", "JEMMIG"))
        gpu = 0

        config = {
            "output_dir": output_dir,
            "save_dir": save_dir,

            "num_samples": num_samples, # 10000,
            # "batch": 20,
            "batch": 10,
            "gpu_support": "cupy",
            "gpu_id": gpu,
        }

        arg_str = get_arg_string(config)
        set_GPUs([gpu])

        print("Running arguments: [{}]".format(arg_str))
        run_command = "{} ./14a_JEMMIG_sampling.py {}".format(PYTHON_EXE, arg_str).strip()
        subprocess.call(run_command, shell=True)
