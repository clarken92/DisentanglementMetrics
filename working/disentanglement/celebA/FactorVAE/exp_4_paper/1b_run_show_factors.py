from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
run_id = "0a_1Konny_multiSave"
# run_id = "0b_1Konny_multiSave"
# run_id = "1_tc50_multiSave"
# run_id = "1a_tc50_multiSave"
# run_id = "1b_tc50_multiSave"
# run_id = "2_tc50_zdim3"
# run_id = "2a_tc50_zdim3"
# run_id = "2b_tc50_zdim3"
# run_id = "3_tc10_zdim3"
# run_id = "3a_tc10_zdim3"
# run_id = "3b_tc10_zdim3"
# run_id = "4_tc50_zdim100"
# run_id = "5_tc50_zdim200"

# enc_dec_model = "my"
# run_id = "0b_my_multiSave"

output_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary", "show_factors"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,
}

arg_str = get_arg_string(config)
set_GPUs([1])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./1a_show_factors.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)