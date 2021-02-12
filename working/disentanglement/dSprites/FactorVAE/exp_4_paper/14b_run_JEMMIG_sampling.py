from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


# Tensorflow version < 1.13 use cuda 9.0
# We should use jax with cuda 9.0
enc_dec_model = "1Konny"

# run_id = "0_FactorVAE_tc10"
# run_id = "0a_FactorVAE_tc10"
# run_id = "0b_FactorVAE_tc10"
# run_id = "0c_FactorVAE_tc10"

# run_id = "1_FactorVAE_tc4"

# run_id = "2_FactorVAE_tc20"
# run_id = "2a_FactorVAE_tc20"
# run_id = "2b_FactorVAE_tc20"
# run_id = "2c_FactorVAE_tc20"

# run_id = "2z_FactorVAE_tc30"

# run_id = "3_FactorVAE_tc50"
# run_id = "3a_FactorVAE_tc50"
# run_id = "3b_FactorVAE_tc50"
# run_id = "3c_FactorVAE_tc50"

# run_id = "4_FactorVAE_tc100"
# run_id = "4a_FactorVAE_tc100"
# run_id = "4b_FactorVAE_tc100"
# run_id = "4c_FactorVAE_tc100"

# run_id = "6_VAE_beta50"
# run_id = "6a_VAE_beta50"
# run_id = "6b_VAE_beta50"
# run_id = "6c_VAE_beta50"

# run_id = "7_VAE_beta1"
# run_id = "7a_VAE_beta1"
# run_id = "7b_VAE_beta1"
# run_id = "7c_VAE_beta1"

# run_id = "8_VAE_beta4"
# run_id = "8a_VAE_beta4"
# run_id = "8b_VAE_beta4"
# run_id = "8c_VAE_beta4"
#
# run_id = "9_VAE_beta10"
# run_id = "9a_VAE_beta10"
# run_id = "9b_VAE_beta10"
# run_id = "9c_VAE_beta10"
#
# run_id = "9z_VAE_beta15"
#
# run_id = "10_VAE_beta20"
# run_id = "10a_VAE_beta20"
# run_id = "10b_VAE_beta20"
# run_id = "10c_VAE_beta20"
#
# run_id = "11_VAE_beta30"
# run_id = "11b_VAE_beta30"

'''
run_ids = [
    "0_FactorVAE_tc10",
    "0a_FactorVAE_tc10",
    "0b_FactorVAE_tc10",
    "0c_FactorVAE_tc10",

    "1_FactorVAE_tc4",

    "2_FactorVAE_tc20",
    "2a_FactorVAE_tc20",
    "2b_FactorVAE_tc20",
    "2c_FactorVAE_tc20",

    "2z_FactorVAE_tc30",

    "3_FactorVAE_tc50",
    "3a_FactorVAE_tc50",
    "3b_FactorVAE_tc50",
    "3c_FactorVAE_tc50",

    "4_FactorVAE_tc100",
    "4a_FactorVAE_tc100",
    "4b_FactorVAE_tc100",
    "4c_FactorVAE_tc100",

    "6_VAE_beta50",
    "6a_VAE_beta50",
    "6b_VAE_beta50",
    "6c_VAE_beta50",

    "7_VAE_beta1",
    "7a_VAE_beta1",
    "7b_VAE_beta1",
    "7c_VAE_beta1",

    "8_VAE_beta4",
    "8a_VAE_beta4",
    "8b_VAE_beta4",
    "8c_VAE_beta4",

    "9_VAE_beta10",
    "9a_VAE_beta10",
    "9b_VAE_beta10",
    "9c_VAE_beta10",

    "9z_VAE_beta15",

    "10_VAE_beta20",
    "10a_VAE_beta20",
    "10b_VAE_beta20",
    "10c_VAE_beta20",

    "11_VAE_beta30",
]
'''

run_ids = [
    # "0_FactorVAE_tc10",
    # "2_FactorVAE_tc20",
    # "2z_FactorVAE_tc30",
    # "3_FactorVAE_tc50",
    # "4_FactorVAE_tc100",
    # "6_VAE_beta50",
    # "8_VAE_beta4",
    # "9_VAE_beta10",
    # "10_VAE_beta20",
    # "11_VAE_beta30",

    # "49b_VAE_beta10_z20",
    # "49e_VAE_beta10_z15",
    # "49a_VAE_beta10_z5",
    "49c_VAE_beta10_z30",
    "49f_VAE_beta10_z12",
    "49g_VAE_beta10_z13",
    "49h_VAE_beta10_z14",
]


for num_samples in [10000, 20000, 50000, 5000, 2000, 1000]:
    for run_id in run_ids:
        output_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", enc_dec_model, run_id))
        save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "JEMMIG"))
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
