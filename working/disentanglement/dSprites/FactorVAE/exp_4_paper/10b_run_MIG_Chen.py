from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"

run_ids = [
    # "0_FactorVAE_tc10",
    # "0a_FactorVAE_tc10",
    # "0b_FactorVAE_tc10",
    # "0c_FactorVAE_tc10",
    #
    # "1_FactorVAE_tc4",
    # "1a_FactorVAE_tc4",
    # "1b_FactorVAE_tc4",
    # "1c_FactorVAE_tc4",
    #
    # "2_FactorVAE_tc20",
    # "2a_FactorVAE_tc20",
    # "2b_FactorVAE_tc20",
    # "2c_FactorVAE_tc20",
    #
    # "2z_FactorVAE_tc30",
    #
    # "3_FactorVAE_tc50",
    # "3a_FactorVAE_tc50",
    # "3b_FactorVAE_tc50",
    "3c_FactorVAE_tc50",
    #
    # "4_FactorVAE_tc100",
    # "4a_FactorVAE_tc100",
    # "4b_FactorVAE_tc100",
    # "4c_FactorVAE_tc100",

    # "6_VAE_beta50",
    # "6a_VAE_beta50",
    # "6b_VAE_beta50",
    # "6c_VAE_beta50",

    # "7_VAE_beta1",
    # "7a_VAE_beta1",
    # "7b_VAE_beta1",
    # "7c_VAE_beta1",

    # "8_VAE_beta4",
    # "8a_VAE_beta4",
    # "8b_VAE_beta4",
    # "8c_VAE_beta4",

    # "9_VAE_beta10",
    # "9a_VAE_beta10",
    # "9b_VAE_beta10",
    # "9c_VAE_beta10",

    # "10_VAE_beta20",
    # "10a_VAE_beta20",
    # "10b_VAE_beta20",
    # "10c_VAE_beta20",

    # "11_VAE_beta30",
]

save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "mig_chen"))
gpu = 0

for run_id in run_ids:
    output_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", enc_dec_model, run_id))

    config = {
        "output_dir": output_dir,
        "save_dir": save_dir,

        "num_samples": 10000,
        "gpu_support": "cupy",
        "gpu_id": gpu,
    }

    arg_str = get_arg_string(config)
    set_GPUs([gpu])

    print("Running arguments: [{}]".format(arg_str))
    run_command = "{} ./10a_MIG_Chen.py {}".format(PYTHON_EXE, arg_str).strip()
    subprocess.call(run_command, shell=True)
