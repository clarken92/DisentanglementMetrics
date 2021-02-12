from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR, RAW_DATA_DIR


# Old
# =============================== #
# run_id = "0_default"
# run_id = "0b_batch64_l1"
# run_id = "1b_like0b_z65"
# run_id = "1c_batch64_z65_bce"
# =============================== #

enc_dec_model = "1Konny"
# run_id = "0a_1Konny_multiSave"
# run_id = "0b_1Konny_multiSave"
# run_id = "1_tc50_multiSave"
# run_id = "1a_tc50_multiSave"
# run_id = "1b_tc50_multiSave"
# run_id = "2_tc50_zdim3"
# run_id = "2a_tc50_zdim3"
# run_id = "3_tc10_zdim3"
# run_id = "3a_tc10_zdim3"
# run_id = "4_tc50_zdim100"
# run_id = "5_tc50_zdim200"
# run_id = "6_VAE_beta50"
run_id = "8_VAE"

# enc_dec_model = "my"
# run_id = "0a_default"
# run_id = "0b_my_multiSave"

output_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", enc_dec_model, run_id))

config = {
    "output_dir": output_dir,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./_test.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)