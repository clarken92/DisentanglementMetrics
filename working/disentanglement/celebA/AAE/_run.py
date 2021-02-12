from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, RAW_DATA_DIR

celebA_root_dir = abspath(join(RAW_DATA_DIR, "ComputerVision", "CelebA"))
output_dir = abspath(join(RESULTS_DIR, "celebA", "AAE"))

config = {
    "output_dir": output_dir,
    "celebA_root_dir": celebA_root_dir,

    # 1Konny
    # =================================== #
    "enc_dec_model": "1Konny",

    "run": "0_Gz10",
    "z_dim": 65,
    "G_loss_z1_gen_coeff": 10,


    # "run": "0_Gz50",
    # "run": "0b_Gz50",
    # "run": "0c_Gz50",
    # "z_dim": 65,
    # "G_loss_z1_gen_coeff": 50,

    # "run": "1_Gz100",
    # "z_dim": 100,
    # "G_loss_z1_gen_coeff": 50,

    # "run": "2_Gz200",
    # "z_dim": 200,
    # "G_loss_z1_gen_coeff": 50,
    # =================================== #

    "force_rm_dir": True,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./_train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)