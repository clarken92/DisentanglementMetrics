from os import environ


def set_GPUs(gpus):
    gpus = [gpus] if not hasattr(gpus, '__len__') else gpus
    print("Using GPU devices: {}".format(gpus))
    environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpus))
    environ['CUDA_DEVICE_IDX'] = str(0)
