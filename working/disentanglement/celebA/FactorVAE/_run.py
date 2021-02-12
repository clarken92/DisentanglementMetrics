from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, RAW_DATA_DIR

celebA_root_dir = abspath(join(RAW_DATA_DIR, "ComputerVision", "CelebA"))
output_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE"))

config = {
    "output_dir": output_dir,
    "celebA_root_dir": celebA_root_dir,

    # 1Konny
    # =================================== #
    "enc_dec_model": "1Konny",

    # FactorVAE
    # ----------------------------------- #
    # "run": "0a_1Konny_multiSave",
    # "run": "0b_1Konny_multiSave",
    # "run": "0c_1Konny_multiSave",
    # "run": "0d_1Konny_multiSave",

    # "run": "1_tc50_multiSave",
    # "run": "1a_tc50_multiSave",
    # "run": "1b_tc50_multiSave",
    # "run": "1c_tc50_multiSave",
    # "tc_loss_coeff": 50,

    # "run": "2_tc50_zdim3",
    # "run": "2a_tc50_zdim3",
    # "run": "2b_tc50_zdim3",
    # "z_dim": 3,
    # "tc_loss_coeff": 50,

    # "run": "3_tc10_zdim3",
    # "run": "3a_tc10_zdim3",
    # "run": "3b_tc10_zdim3",
    # "z_dim": 3,
    # "tc_loss_coeff": 10,

    # "run": "4_tc50_zdim100",
    # "z_dim": 100,
    # "tc_loss_coeff": 50,

    # "run": "5_tc50_zdim200",
    # "z_dim": 200,
    # "tc_loss_coeff": 50,

    # "run": "7_zdim10",
    # "run": "7a_zdim10",
    # "z_dim": 10,
    # "tc_loss_coeff": 50,
    
    # Beta-VAE
    # ----------------------------------- #
    "lr_Dz": 0,
    "Dz_tc_loss_coeff": 0,
    "tc_loss_coeff": 0,
    "gp0_z_tc_coeff": 0,

    "run": "7_VAE_beta10",
    "kld_loss_coeff": 10,

    # "run": "6_VAE_beta50",
    # "run": "6a_VAE_beta50",
    # "run": "6b_VAE_beta50",
    # "run": "6c_VAE_beta50",
    # "kld_loss_coeff": 50,
    # ----------------------------------- #


    # VAE
    # ----------------------------------- #
    # "run": "8_VAE",
    # "z_dim": 65,

    # "run": "9_VAE_zdim100",
    # "z_dim": 100,

    # "run": "10_VAE_zdim200",
    # "z_dim": 200,

    # "lr_Dz": 0,
    # "Dz_tc_loss_coeff": 0,
    # "tc_loss_coeff": 0,
    # "gp0_z_tc_coeff": 0,
    # "kld_loss_coeff": 1,
    # ----------------------------------- #
    # =================================== #

    "force_rm_dir": True,
}

arg_str = get_arg_string(config)
set_GPUs([1])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./_train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)