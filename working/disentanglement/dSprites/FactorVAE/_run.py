from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, RAW_DATA_DIR


output_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE"))

config = {
    "output_dir": output_dir,
    "enc_dec_model": "1Konny",

    # "run": "0_l1",
    # "rec_x_mode": "l1",

    # "run": "0a_mse",
    # "rec_x_mode": "mse",

    # "run": "0b_bec",
    # "rec_x_mode": "bce",

    # "run": "1_bce_batch_64",
    # "rec_x_mode": "bce",

    # FactorVAE
    # ----------------------------------- #
    # "run": "0_FactorVAE_tc10",
    # "run": "0a_FactorVAE_tc10",
    # "run": "0b_FactorVAE_tc10",
    # "run": "0c_FactorVAE_tc10",
    # "tc_loss_coeff": 10,

    # "run": "1_FactorVAE_tc4",
    # "run": "1a_FactorVAE_tc4",
    # "run": "1b_FactorVAE_tc4",
    # "run": "1c_FactorVAE_tc4",
    # "tc_loss_coeff": 4,

    # "run": "2_FactorVAE_tc20",
    # "run": "2a_FactorVAE_tc20",
    # "run": "2b_FactorVAE_tc20",
    # "run": "2c_FactorVAE_tc20",
    # "tc_loss_coeff": 20,

    # "run": "2z_FactorVAE_tc30",
    # "tc_loss_coeff": 30,

    # "run": "3_FactorVAE_tc50",
    # "run": "3a_FactorVAE_tc50",
    # "run": "3b_FactorVAE_tc50",
    # "run": "3c_FactorVAE_tc50",
    # "tc_loss_coeff": 50,

    # "run": "4_FactorVAE_tc100",
    # "run": "4a_FactorVAE_tc100",
    # "run": "4b_FactorVAE_tc100",
    # "run": "4c_FactorVAE_tc100",
    # "tc_loss_coeff": 100,
    # ----------------------------------- #

    # Beta - VAE
    # ----------------------------------- #
    "lr_Dz": 0,
    "Dz_tc_loss_coeff": 0,
    "tc_loss_coeff": 0,
    "gp0_z_tc_coeff": 0,

    "run": "6_VAE_beta50",
    # "run": "6a_VAE_beta50",
    # "run": "6b_VAE_beta50",
    # "run": "6c_VAE_beta50",
    "kld_loss_coeff": 50,

    # "run": "7_VAE_beta1",
    # "run": "7a_VAE_beta1",
    # "run": "7b_VAE_beta1",
    # "run": "7c_VAE_beta1",
    # "kld_loss_coeff": 1,

    # "run": "8_VAE_beta4",
    # "run": "8a_VAE_beta4",
    # "run": "8b_VAE_beta4",
    # "run": "8c_VAE_beta4",
    # "kld_loss_coeff": 4,

    # "run": "9_VAE_beta10",
    # "run": "9a_VAE_beta10",
    # "run": "9b_VAE_beta10",
    # "run": "9c_VAE_beta10",
    # "kld_loss_coeff": 10,
    
    # "run": "9z_VAE_beta15",
    # "kld_loss_coeff": 15,
    
    # "run": "10_VAE_beta20",
    # "run": "10a_VAE_beta20",
    # "run": "10b_VAE_beta20",
    # "run": "10c_VAE_beta20",
    # "kld_loss_coeff": 20,

    # "run": "10z_VAE_beta25",
    # "kld_loss_coeff": 25,

    # "run": "11_VAE_beta30",
    # "run": "11b_VAE_beta30",
    # "kld_loss_coeff": 30,
    # ----------------------------------- #

    "force_rm_dir": True,
}

arg_str = get_arg_string(config)
set_GPUs([3])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./_train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)