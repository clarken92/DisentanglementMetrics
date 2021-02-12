import tensorflow as tf

from ..shaping import mixed_shape


def central_crop_with_fixed_size(image, target_height, target_width):
    image = tf.convert_to_tensor(image, name='image')

    is_batch = True
    ndims = image.get_shape().ndims

    if ndims is None:
        is_batch = False
        image = tf.expand_dims(image, 0)
        image.set_shape([None] * 4)
    elif ndims == 3:
        is_batch = False
        image = tf.expand_dims(image, 0)
    elif ndims != 4:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    image_shape = mixed_shape(image)
    assert len(image_shape) == 4
    input_height, input_width = image_shape[1], image_shape[2]

    offset_height = (input_height - target_height) // 2
    offset_width = (input_width - target_width) // 2

    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_height=offset_height, offset_width=offset_width,
        target_height=target_height, target_width=target_width)

    if not is_batch:
        cropped_image = tf.squeeze(cropped_image, axis=[0])

    return cropped_image