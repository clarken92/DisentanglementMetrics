from os.path import join, exists, abspath
from os import makedirs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from global_settings import RESULTS_DIR


def plot_curve():
    result_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary",
                              "learning_curves"))
    save_dir = result_dir
    if not exists(save_dir):
        makedirs(save_dir)

    file_names = ['run_1Konny_0_default_summary_tf-tag-train_tc_loss.csv',
                  'run_1Konny_1_tc50_multiSave_summary_tf-tag-train_tc_loss.csv',
                  ]
    file_paths = [join(result_dir, file_name) for file_name in file_names]

    labels = ['TC=10',
              'TC=50',
              ]

    assert len(labels) == len(file_paths)

    xs = [None] * len(labels)
    ys = [None] * len(labels)

    # Load to Numpy files
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path, header=0, sep=',')

        x = df.as_matrix(columns=["Step"]).astype(np.int32)
        if len(x.shape) == 2:
            assert x.shape[-1] == 1
            x = x.reshape(x.shape[0])
        assert len(x.shape) == 1

        y = df.as_matrix(columns=["Value"]).astype(np.float32)
        if len(y.shape) == 2:
            assert y.shape[-1] == 1
            y = y.reshape(x.shape[0])
        assert len(y.shape) == 1

        xs[i] = x
        ys[i] = y

    min_common_len = min([len(x) for x in xs])
    common_x = xs[0][: min_common_len]

    font = {'family': 'normal', 'size': 14}
    matplotlib.rc('font', **font)

    # Plot curves
    for y, label in zip(ys, labels):
        plt.plot(common_x, y[:min_common_len], label="{}".format(label), linewidth=2.0)

    x_ticks = list(range(0, 350000, 50000))
    plt.xticks(ticks=x_ticks, labels=['{:}k'.format(int(round(1.0 * x / 1000))) for x in x_ticks])
    plt.xlim(xmin=-4000, xmax=254000)  # set here

    plt.ylabel("TC loss")
    plt.xlabel("step")

    plt.grid(alpha=0.5)
    plt.legend(loc="upper right")
    plt.subplots_adjust(**{'left': 0.12, 'right': 0.95, 'bottom': 0.13, 'top': 0.98})

    pp = PdfPages(join(save_dir, 'tc_curve.pdf'))
    pp.savefig(plt.gcf())
    pp.close()
    plt.close()


def main():
    plot_curve()


if __name__ == "__main__":
    main()
