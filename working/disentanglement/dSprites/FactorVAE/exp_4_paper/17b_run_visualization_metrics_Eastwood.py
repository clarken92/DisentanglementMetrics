from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from my_utils.tensorflow_utils.training import set_GPUs
from global_settings import PYTHON_EXE, RESULTS_DIR


enc_dec_model = "1Konny"
run_id = "3b_FactorVAE_tc50"
# run_id = "4b_FactorVAE_tc100"

output_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", enc_dec_model, run_id))
save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "metrics_Eastwood_plot"))
metrics_Eastwood_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "metrics_Eastwood"))

config = {
    "output_dir": output_dir,
    "save_dir": save_dir,

    "metrics_Eastwood_dir": metrics_Eastwood_dir,
    "continuous_only": True,
    "LASSO_alpha": 0.002,
    "LASSO_iters": 10000,
}

arg_str = get_arg_string(config)
set_GPUs([0])

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./17a_visualization_metrics_Eastwood.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)