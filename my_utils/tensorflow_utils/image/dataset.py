import os
import functools
import tensorflow as tf
import numpy as np

from .transformation import central_crop_with_fixed_size


# Load raw images from folder. Only support placeholder
# ================================================== #
class BaseImageFolderDatasetLoader(object):
    def __init__(self, dir_path, img_formats=None, excluded_img_names=None):
        assert os.path.isdir(dir_path), "'{}' is not an existed folder!".format(dir_path)
        self.dir_path = dir_path

        supported_formats = {'png', 'jpg', 'jpeg'}

        if img_formats is None:
            img_formats = supported_formats
        elif isinstance(img_formats, str):
            if img_formats == "jpg" or img_formats == "jpeg":
                img_formats = {"jpg", "jpeg"}
            else:
                img_formats = {img_formats}
        elif isinstance(img_formats, (list, tuple)):
            img_formats = set(img_formats)

        assert isinstance(img_formats, set), "'img_formats' must be None, a string, or " \
            "a list/tuple/set of strings. Found {}!".format(type(img_formats))
        assert len(set(img_formats).difference(supported_formats)) == 0, \
            "'img_formats' contain unsupported format. Only support {}!".format(supported_formats)

        self.img_formats = list(img_formats)

        if excluded_img_names is None:
            img_names = [file_name for file_name in os.listdir(self.dir_path) if self._is_supported(file_name)]
        else:
            assert hasattr(excluded_img_names, '__len__')
            img_names = [file_name for file_name in os.listdir(self.dir_path) if
                (self._is_supported(file_name) and file_name not in excluded_img_names)]

        assert len(img_names) > 0, "There are no valid '{}' images in '{}'".format(self.img_formats, dir_path)
        self.img_files = [os.path.join(self.dir_path, img_name) for img_name in img_names]

    # def _is_supported(self, file_name):
    #     return file_name.endswith(self.img_ext)

    def _is_supported(self, file_name):
        result = False
        for ext in self.img_formats:
            result = result or file_name.endswith(ext)
        return result

    @property
    def num_data(self):
        return len(self.img_files)

    def sample_files(self, batch_ids):
        return [self.img_files[idx] for idx in batch_ids]


class TFImageFolderDatasetLoader(BaseImageFolderDatasetLoader):
    def __init__(self, dir_path, img_formats=None):
        assert isinstance(img_formats, str), "'img_formats' must be a string!"
        super(TFImageFolderDatasetLoader, self).__init__(dir_path, img_formats)
        self.img_formats = img_formats

        self.image_files_ph = tf.placeholder(dtype=tf.string, shape=[None], name="image_files_ph")
        self.transformed_img = None

    def build_transformation_flow_tf(self, *transform_fns):
        # 'transform_fns': List of image transformation functions in Tensorflow

        # dataset = tf.data.Dataset.from_tensor_slices(self.image_files_ph)
        # dataset = dataset.map(tf.read_file, num_parallel_calls=10)
        # iterator = dataset.make_one_shot_iterator()

        if self.img_formats == "png":
            read_image_fn = lambda img_file: tf.image.decode_png(tf.read_file(img_file))
        elif self.img_formats == "jpg":
            read_image_fn = lambda img_file: tf.image.decode_jpeg(tf.read_file(img_file))
        else:
            raise ValueError("Do not support image extension '{}'!".format(self.img_formats))

        img = tf.map_fn(read_image_fn, self.image_files_ph, dtype=tf.uint8,
                        parallel_iterations=20, back_prop=False,
                        infer_shape=True, swap_memory=True)

        for transform_fn in transform_fns:
            img = transform_fn(img)

        self.transformed_img = img

    def sample_images(self, sess, batch_ids):
        assert self.transformed_img is not None, "'build_transformation_flow_tf()' must be called in advance!"

        b_img_files = self.sample_files(batch_ids)
        return sess.run(self.transformed_img, feed_dict={self.image_files_ph: b_img_files})

    def sample_images_by_names(self, sess, img_names):
        b_img_files = [os.path.join(self.dir_path, img_name) for img_name in img_names]
        return sess.run(self.transformed_img, feed_dict={self.image_files_ph: b_img_files})


class TFCelebALoader(TFImageFolderDatasetLoader):
    def __init__(self, root_dir):
        img_dir = os.path.join(root_dir, "Img", "img_align_celeba")
        super(TFCelebALoader, self).__init__(img_dir, img_formats="jpg")

        train_test_split_file = os.path.join(root_dir, "Eval", "list_eval_partition.txt")
        train_img_files = []
        valid_img_files = []
        test_img_files = []

        with open(train_test_split_file, 'r') as f:
            for line in f.readlines():
                file_name, tag = line.strip().split(" ")
                tag = int(tag)

                if tag == 0:
                    train_img_files.append(os.path.join(img_dir, file_name))
                elif tag == 1:
                    valid_img_files.append(os.path.join(img_dir, file_name))
                elif tag == 2:
                    test_img_files.append(os.path.join(img_dir, file_name))
                else:
                    raise ValueError("Do not support tag={}!".format(tag))

        assert len(train_img_files) + len(test_img_files) + len(valid_img_files) == len(self.img_files)

        self.train_img_files = train_img_files
        self.test_img_files = test_img_files
        self.valid_img_files = valid_img_files

    def sample_files(self, batch_ids):
        raise NotImplementedError

    def sample_images(self, sess, batch_ids):
        raise NotImplementedError

    def sample_files_from_dataset(self, dataset_type, batch_ids):
        if dataset_type == "train":
            img_files = self.train_img_files
        elif dataset_type == "valid":
            img_files = self.valid_img_files
        elif dataset_type == "test":
            img_files = self.test_img_files
        else:
            raise ValueError("Only support 'train', 'valid', 'test' datasets!")

        return [img_files[idx] for idx in batch_ids]

    def sample_images_from_dataset(self, sess, dataset_type, batch_ids):
        assert self.transformed_img is not None, "'build_transformation_flow_tf()' must be called in advance!"

        b_img_files = self.sample_files_from_dataset(dataset_type, batch_ids)
        return sess.run(self.transformed_img, feed_dict={self.image_files_ph: b_img_files})

    @property
    def num_train_data(self):
        return len(self.train_img_files)

    @property
    def num_valid_data(self):
        return len(self.valid_img_files)

    @property
    def num_test_data(self):
        return len(self.test_img_files)

    @staticmethod
    def get_transform_fns(name, verbose=True, **kwargs):
        possible_names = ["carpedm20", "1Konny"]
        # In DCGAN, it is 108, in BEGAN, it is 128
        if name == "carpedm20":
            crop_size = kwargs.get('crop_size', (128, 128))
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)

            resize_size = kwargs.get('resize_size', (64, 64))
            if isinstance(resize_size, int):
                resize_size = (resize_size, resize_size)

            fns = [
                lambda a: tf.cast(a, dtype=tf.float32) / 255.0,  # Convert image to [0, 1]
                functools.partial(central_crop_with_fixed_size,
                                  target_height=crop_size[0], target_width=crop_size[1]),
                functools.partial(tf.image.resize_images, size=resize_size),
            ]

            if verbose:
                print("Flow of transformations by '{}':".format(name))
                print("Normalize images to [0, 1]")
                print("Central crop images to size {}".format(tuple(crop_size)))
                print("Resize images to size {}".format(tuple(resize_size)))

        elif name == "1Konny":
            resize_size = kwargs.get('resize_size', (64, 64))
            if isinstance(resize_size, int):
                resize_size = (resize_size, resize_size)

            fns = [
                lambda a: tf.cast(a, dtype=tf.float32) / 255.0,  # Convert image to [0, 1]
                functools.partial(tf.image.resize_images, size=resize_size),
            ]

            if verbose:
                print("Flow of transformations by '{}':".format(name))
                print("Normalize images to [0, 1]")
                print("Resize images to size {}".format(tuple(resize_size)))

        else:
            raise ValueError("Only support one of the following names: {}".format(possible_names))

        return fns


class TFCelebAWithAttrLoader(TFCelebALoader):
    def __init__(self, root_dir):
        super(TFCelebAWithAttrLoader, self).__init__(root_dir)

        attr_file = os.path.join(root_dir, "Anno", "list_attr_celeba.txt")
        with open(attr_file, "r") as f:
            f.readline().strip()

            line = f.readline().strip()
            attr_names = line.split()
            self.attr_names = attr_names  # 40 attributes
            self.attr_name2attr_idx = {name: i for i, name in enumerate(attr_names)}

            img_names = []
            img_attrs = []

            count = 2
            for line in f:
                line = line.strip()
                values = line.split()
                assert len(values) == 41, "At line {} (0 index), " \
                    "len(values)={}, values={}".format(count, len(values), values)

                img_names.append(values[0])
                img_attrs.append(values[1:])

                count += 1

            assert len(img_names) == len(img_attrs) == self.num_data, \
                "len(img_names)={}, len(img_attrs)={}, self.num_data={}".format(
                    len(img_names), len(img_attrs), self.num_data)

            img_attrs = (np.asarray(img_attrs, dtype=np.int32) == 1)
            self.img_attrs = img_attrs

            train_attr_ids = []
            valid_attr_ids = []
            test_attr_ids = []

            for i, img_file in enumerate(self.train_img_files):
                attr_idx = os.path.splitext(os.path.basename(img_file))[0]
                attr_idx = int(attr_idx) - 1
                assert img_file.endswith(img_names[attr_idx]), "At attr_idx={}, image name is '{}' " \
                    "while train image file {} is '{}'!".format(attr_idx, img_names[attr_idx], i, img_file)
                train_attr_ids.append(attr_idx)

            for i, img_file in enumerate(self.valid_img_files):
                attr_idx = os.path.splitext(os.path.basename(img_file))[0]
                attr_idx = int(attr_idx) - 1
                assert img_file.endswith(img_names[attr_idx]), "At attr_idx={}, image name is '{}' " \
                    "while valid image file {} is '{}'!".format(attr_idx, img_names[attr_idx], i, img_file)
                valid_attr_ids.append(attr_idx)

            for i, img_file in enumerate(self.test_img_files):
                attr_idx = os.path.splitext(os.path.basename(img_file))[0]
                attr_idx = int(attr_idx) - 1
                assert img_file.endswith(img_names[attr_idx]), "At attr_idx={}, image name is '{}' " \
                    "while test image file {} is '{}'!".format(attr_idx, img_names[attr_idx], i, img_file)
                test_attr_ids.append(attr_idx)

            self.train_attr_ids = np.asarray(train_attr_ids, dtype=np.int32)
            self.valid_attr_ids = np.asarray(valid_attr_ids, dtype=np.int32)
            self.test_attr_ids = np.asarray(test_attr_ids, dtype=np.int32)

    def sample_attrs_from_dataset(self, dataset_type, batch_ids, attr_names=None):
        if dataset_type == "train":
            attr_sample_ids = self.train_attr_ids[batch_ids]
        elif dataset_type == "valid":
            attr_sample_ids = self.valid_attr_ids[batch_ids]
        elif dataset_type == "test":
            attr_sample_ids = self.test_attr_ids[batch_ids]
        else:
            raise ValueError("Only support 'train', 'valid', 'test' datasets!")

        if not attr_names:
            results = self.img_attrs[attr_sample_ids, :]
        else:
            if isinstance(attr_names, str):
                attr_names = [attr_names]

            attr_comp_ids = [self.attr_name2attr_idx[name] for name in attr_names]

            results = self.img_attrs[attr_sample_ids, :]
            results = results[:, attr_comp_ids]

        return results

    @property
    def attributes(self):
        return self.attr_names

    @property
    def num_attributes(self):
        return len(self.attr_names)

    def get_attr_comp_ids(self, attr_names):
        assert isinstance(attr_names, (list, tuple)), "'attr_names' must be a list!"
        return [self.attr_name2attr_idx[name] for name in attr_names]