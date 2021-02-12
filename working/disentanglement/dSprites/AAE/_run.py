from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, RAW_DATA_DIR


output_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE"))

config = {
    "output_dir": output_dir,
    "enc_dec_model": "1Konny",

    "run": "0_Gz50",
    # "run": "0b_Gz50",
    # "run": "0c_Gz50",
    "G_loss_z1_gen_coeff": 50,
    "z_dim": 65,

    # "run": "1_Gz50_zdim10",
    # "run": "1a_Gz50_zdim10",
    # "run": "1b_Gz50_zdim10",
    # "run": "1c_Gz50_zdim10",
    # "G_loss_z1_gen_coeff": 50,

    # "run": "2_Gz20_zdim10",
    # "run": "2a_Gz20_zdim10",
    # "run": "2b_Gz20_zdim10",
    # "run": "2c_Gz20_zdim10",
    # "G_loss_z1_gen_coeff": 20,

    # "run": "3_Gz10_zdim10",
    # "run": "3a_Gz10_zdim10",
    # "run": "3b_Gz10_zdim10",
    # "run": "3c_Gz10_zdim10",
    # "G_loss_z1_gen_coeff": 10,

    # "z_dim": 10,

    "force_rm_dir": True,
}

arg_str = get_arg_string(config)
set_GPUs([1])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./_train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)