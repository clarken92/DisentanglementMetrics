import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def central_crop_with_fixed_size(image, target_height, target_width):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 2 or image.ndim == 3, \
        "'image' must be a 2D (grayscale) or 3D (color) array!"

    input_height, input_width = image.shape[0], image.shape[1]

    assert (target_height <= input_height), \
        "'target_height' is greater than image height ({} and {})".format(
            target_height, input_height)

    assert (target_width <= input_width), \
        "'target_width' is greater than image width ({} and {})".format(
            target_width, input_width)

    offset_height = (input_height - target_height) // 2
    offset_width = (input_width - target_width) // 2

    cropped_image = image[offset_height: offset_height + target_height,
                    offset_width: offset_width + target_width]

    return cropped_image


# Convert pixel values
# --------------------------------------------- #
def float_to_uint8(x, pixel_inv_scale, pixel_shift):
    # pixel_inv_scale = 255, pixel_shift = 0: [0, 1] => [0, 255]
    # pixel_inv_scale = 127.5, pixel_shift = -1: [-1, 1] => [0, 255]
    assert x.dtype == np.float32, "x.dtype={}".format(x.dtype)
    return np.clip((x - pixel_shift) * pixel_inv_scale, 0.0, 255.0).astype(np.uint8)


def uint8_to_float(x, pixel_inv_scale, pixel_shift):
    # pixel_inv_scale = 255, pixel_shift = 0: [0, 255] => [0, 1]
    # pixel_inv_scale = 127.5, pixel_shift = -1: [0, 255] => [-1, 1]
    assert x.dtype == np.uint8, "x.dtype={}".format(x.dtype)
    x = np.asarray(x, dtype=np.float32)
    return x * 1.0 / pixel_inv_scale + pixel_shift


def binary_float_to_uint8(x):
    return float_to_uint8(x, pixel_inv_scale=255, pixel_shift=0)


def m1p1_float_to_uint8(x):
    return float_to_uint8(x, pixel_inv_scale=127.5, pixel_shift=-1)


def uint8_to_binary_float(x):
    return uint8_to_float(x, pixel_inv_scale=255, pixel_shift=0)


def uint8_to_m1p1_float(x):
    return uint8_to_float(x, pixel_inv_scale=127.5, pixel_shift=-1)


def binary_float_to_m1p1_float(x):
    assert x.dtype == np.float32, "x.dtype={}".format(x.dtype)
    # From [0, 1] to [-1, 1]
    return 2.0 * x - 1.0


def m1p1_float_to_binary_float(x):
    assert x.dtype == np.float32, "x.dtype={}".format(x.dtype)
    # From [-1, 1] to [0, 1]
    return (x + 1.0) * 0.5


def scale_to_01(x, axis=None):
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    z = (x - x_min) / (x_max - x_min)
    return z
# --------------------------------------------- #


# Save images
# --------------------------------------------- #
def pack_img_block_array(img_block_array):
    assert len(img_block_array.shape) == 5, "'img_block_array' must be a 5D array of shape " \
                                            "(rows, cols, height, width, channels)! Found {}".format(
        img_block_array.shape)

    # (rows, height, cols, width, channels)
    img_block_array = np.swapaxes(img_block_array, 1, 2)
    shape = list(img_block_array.shape)

    if shape[-1] == 1:
        # (rows * height, cols * width)
        img_block_array = np.reshape(img_block_array, [shape[0] * shape[1], shape[2] * shape[3]])
    else:
        # (rows * height, cols * width, channels)
        img_block_array = np.reshape(img_block_array, [shape[0] * shape[1], shape[2] * shape[3], shape[4]])
    return img_block_array


def pack_imgs_into_blocks(images, num_cols=10, pad_size=2):
    assert len(images.shape) == 4, "'images' must be a 4D array of shape " \
                                   "(batch, height, width, channels)! Found images.shape={}".format(images.shape)

    num_images, height, width, channels = images.shape

    num_rows = num_images // num_cols
    remaining = num_images - (num_rows * num_cols)
    if remaining > 0:
        num_rows += 1

    assert num_rows > 0, f"'num_rows'={num_rows} is not > 0!"
    assert num_cols > 0, f"'num_cols'={num_cols} is not > 0!"

    output = np.zeros([num_rows * (height + pad_size),
                       num_cols * (width + pad_size), channels],
                      dtype=images.dtype)

    for n in range(num_images):
        row_id = n // num_cols
        col_id = n % num_cols

        row_start = row_id * (height + pad_size)
        col_start = col_id * (width + pad_size)

        output[row_start: row_start + height, col_start: col_start + width] = images[n]

    output = output[:-pad_size, :-pad_size]
    return output


def highlight_img_block_array(img_block_array, hl_blocks, hl_color='grey', hl_width=1):
    assert img_block_array.dtype == np.uint8, "'img_block_array' must have 'unit8' type!"
    assert len(img_block_array.shape) == 5, "'img_block_array' must be a 5D array of shape " \
                                            "(rows, cols, height, width, channels)! Found {}".format(
        img_block_array.shape)
    height, width, channel = img_block_array.shape[2], img_block_array.shape[3], img_block_array.shape[4]

    color_map = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'black': [0],
        'grey': [127],
        'white': [255]
    }

    if isinstance(hl_color, str):
        hl_color = color_map.get(hl_color, hl_color)
    hl_color = np.asarray(hl_color, dtype=np.uint8)

    assert len(hl_color.shape) == 1 and (hl_color.shape[0] == 1 or hl_color.shape[0] == 3), \
        "'hl_color' must be a recognized string or 1D array of length 1 or 3. Found {}!".format(hl_color)

    if channel == 1 and len(hl_color) == 3:
        img_block_array = np.tile(img_block_array, [1, 1, 1, 1, 3])

    hl_color = np.expand_dims(np.expand_dims(hl_color, axis=0), axis=0)

    for row_idx, col_idx in hl_blocks:
        img_block_array[row_idx, col_idx, 0, :, :] = hl_color
        img_block_array[row_idx, col_idx, height - 1, :, :] = hl_color
        img_block_array[row_idx, col_idx, :, 0, :] = hl_color
        img_block_array[row_idx, col_idx, :, width - 1, :] = hl_color

    return img_block_array


def save_img(save_file, img):
    assert img.dtype == np.uint8, "'img' must have 'unit8' type!"
    img = Image.fromarray(img)
    img.save(save_file)


def save_img_block(save_file, img_block_array):
    # Save a single image block
    assert save_file.endswith(".png") or save_file.endswith(".jpg"), \
        "'save_file' must end with '.png' or '.jpg'!"

    assert img_block_array.dtype == np.uint8, "'img_block_array' must have 'unit8' type!"

    img = Image.fromarray(pack_img_block_array(img_block_array))
    img.save(save_file)


def save_img_block_highlighted(
        save_file, img_block_array, hl_blocks, hl_color="red", hl_width=1):
    # Save a single image block
    assert save_file.endswith(".png") or save_file.endswith(".jpg"), \
        "'save_file' must end with '.png' or '.jpg'!"

    assert img_block_array.dtype == np.uint8, "'img_block_array' must have 'unit8' type!"

    img_block_array = highlight_img_block_array(
        img_block_array, hl_blocks, hl_color=hl_color, hl_width=hl_width)
    img = Image.fromarray(pack_img_block_array(img_block_array))
    img.save(save_file)


def save_img_block_with_ticklabels(
        save_file, img_block_array, cmap="binary",
        x_tick_labels=None, y_tick_labels=None,
        font_size=12, title="", title_font_scale=1.5,
        subplot_adjust={}, size_inches=None):
    assert img_block_array.dtype == np.uint8, "'img_block_array' must have 'unit8' type!"
    assert len(img_block_array.shape) == 5, "'img_block_array' must be a 5D array of shape " \
                                            "(rows, cols, height, width, channels)! Found {}".format(
        img_block_array.shape)

    num_rows, num_cols, img_height, img_width = img_block_array.shape[:4]
    img_array = pack_img_block_array(img_block_array)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_array, cmap=cmap, aspect="equal")

    if x_tick_labels is not None:
        assert hasattr(x_tick_labels, "__len__") and len(x_tick_labels) == num_cols
        plt.xticks([(i + 0.5) * img_width for i in range(num_cols)], x_tick_labels,
                   rotation=45, ha="left", rotation_model="anchor", fontsize=font_size)
    else:
        plt.xticks([])

    if y_tick_labels is not None:
        assert hasattr(y_tick_labels, "__len__") and len(y_tick_labels) == num_rows
        plt.yticks([(i + 0.5) * img_height for i in range(num_rows)], y_tick_labels, fontsize=font_size)
    else:
        plt.yticks([])

    if title:
        plt.title(title, fontsize=int(title_font_scale * font_size))

    plt.subplots_adjust(**subplot_adjust)
    if size_inches is not None:
        plt.gcf().set_size_inches(size_inches)

    if save_file.endswith('.pdf'):
        with PdfPages(save_file) as pdf_file:
            plt.savefig(pdf_file, format='pdf')
            plt.close()
    else:
        plt.savefig(save_file, dpi=300)

    plt.close(fig)


def save_img_block_highlighted_with_ticklabels(
        save_file, img_block_array,
        hl_blocks, hl_color="red", hl_width=1,
        cmap="binary", x_tick_labels=None, y_tick_labels=None,
        font_size=12, title="", title_font_scale=1.5,
        subplot_adjust={}, size_inches=None):
    num_rows, num_cols, img_height, img_width = img_block_array.shape[:4]

    img_block_array = highlight_img_block_array(
        img_block_array, hl_blocks, hl_color=hl_color, hl_width=hl_width)
    img_array = pack_img_block_array(img_block_array)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_array, cmap=cmap, aspect="equal")

    # Set x_tick
    if x_tick_labels is not None:
        assert hasattr(x_tick_labels, "__len__") and len(x_tick_labels) == num_cols
        plt.xticks([(i + 0.5) * img_width for i in range(num_cols)], x_tick_labels,
                   rotation=45, ha="left", rotation_model="anchor", fontsize=font_size)
    else:
        plt.xticks([])

    # Set y_tick
    if y_tick_labels is not None:
        assert hasattr(y_tick_labels, "__len__") and len(y_tick_labels) == num_rows
        plt.yticks([(i + 0.5) * img_height for i in range(num_rows)], y_tick_labels, fontsize=font_size)
    else:
        plt.yticks([])

    if title:
        plt.title(title, fontsize=int(title_font_scale * font_size))

    plt.subplots_adjust(**subplot_adjust)
    if size_inches is not None:
        plt.gcf().set_size_inches(size_inches)

    if save_file.endswith('.pdf'):
        with PdfPages(save_file) as pdf_file:
            plt.savefig(pdf_file, format='pdf')
            plt.close()
    else:
        plt.savefig(save_file, dpi=300)

    plt.close(fig)


def save_img_blocks_col_by_col(save_file, img_blocks):
    # Save image blocks column by column

    assert save_file.endswith(".png") or save_file.endswith(".jpg"), \
        "'save_file' must end with '.png' or '.jpg'!"

    assert isinstance(img_blocks, (list, tuple)), "'img_blocks' must be a list/tuple!"

    for n in range(len(img_blocks)):
        assert img_blocks[n].dtype == np.uint8, "'img_block[{}]' must be 'unit8'!".format(n)
        assert len(img_blocks[n].shape) == 5, "'img_blocks[{}]' must be a 5D array of shape " \
                                              "(rows, cols, height, width, channels)!. Found {}".format(n, img_blocks[
            n].shape)
        if img_blocks[n].shape[-1] == 1:
            img_blocks[n] = np.reshape(img_blocks[n], img_blocks[n].shape[:-1])

    img_block_shape = img_blocks[0].shape
    n_row = img_block_shape[0]
    n_col = img_block_shape[1]

    # (n_row, n_col, height, width, channel)
    big_img_block = np.zeros([n_row, n_col * len(img_blocks)] + list(img_block_shape[2:]), dtype=np.uint8)
    for n in range(len(img_blocks)):
        assert img_blocks[n].shape == img_block_shape, "'img_blocks[{}]' " \
                                                       "does not have the same shape as img_blocks[0]".format(n)

        for i in range(n_col):
            big_img_block[:, i * len(img_blocks) + n, ...] = img_blocks[n][:, i, ...]

    # x: (n_row, height, n_col, width, channel)
    big_img_block = np.swapaxes(big_img_block, 1, 2)
    shape = list(big_img_block.shape)
    # x: (n_row * height, n_col * width, channel)
    big_img_block = np.reshape(big_img_block, [shape[0] * shape[1], shape[2] * shape[3]] + shape[4:])

    img = Image.fromarray(big_img_block)
    img.save(save_file)
# --------------------------------------------- #


def merge_batch_img_along_width(batch_img):
    assert len(batch_img.shape) == 4, "'batch_imgs' must be a 4D array of shape " \
                                      "(batch, height, width, channels)!. Found {}".format(batch_img.shape)
    # (height, batch, width, channels)
    batch_img = np.transpose(batch_img, [1, 0, 2, 3])
    shape = batch_img.shape
    # (height, batch * width, channels)
    batch_img = np.reshape(batch_img, [shape[0], shape[1] * shape[2], shape[3]])
    return batch_img


def merge_batch_img_along_height(batch_img):
    assert len(batch_img.shape) == 4, "'batch_img' must be a 4D array of shape " \
                                      "(batch, height, width, channels)!. Found {}".format(batch_img.shape)
    shape = batch_img.shape
    # (batch * height, width, channels)
    batch_img = np.reshape(batch_img, [shape[0] * shape[1], shape[2], shape[3]])
    return batch_img
