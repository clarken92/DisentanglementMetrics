import numpy as np
import tensorflow as tf

from my_utils.tensorflow_utils.models import LiteBaseModel
from my_utils.tensorflow_utils.shaping import mixed_shape, flatten_right_from
from my_utils.python_utils.general import to_list, reshape_axes, get_meshgrid
from my_utils.python_utils.image import save_img_block, save_img_block_highlighted, \
    save_img_block_highlighted_with_ticklabels, save_img_block_with_ticklabels, \
    save_img_blocks_col_by_col
from my_utils.python_utils.training import iterate_data


class BaseLatentModel(LiteBaseModel):
    def __init__(self, x_shape, z_shape):
        super(BaseLatentModel, self).__init__()

        self.x_shape = to_list(x_shape)
        self.z_shape = to_list(z_shape)
        self.x_ph = tf.placeholder(tf.float32, [None] + self.x_shape, name="x")
        self.z_ph = tf.placeholder(tf.float32, [None] + self.z_shape, name="z")

    @property
    def _placeholder_4_z(self):
        return self.z_ph
    
    def encode(self, sess, x, **kwargs):
        z1_gen = self.output_dict['z1_gen']
        return sess.run(z1_gen, feed_dict={self.is_train: False, self.x_ph: x})

    def decode(self, sess, z, **kwargs):
        x1_gen = self.output_dict['x1_gen']
        return sess.run(x1_gen, feed_dict={self.is_train: False, self._placeholder_4_z: z})

    def reconstruct(self, sess, x, **kwargs):
        x1 = self.output_dict['x1']
        return sess.run(x1, feed_dict={self.is_train: False, self.x_ph: x})

    def generate_images(self, save_file, sess, z, block_shape,
                        batch_size=20, dec_output_2_img_func=None, **kwargs):
        if batch_size < 0:
            x1_gen = self.decode(sess, z, **kwargs)
        else:
            x1_gen = []
            for batch_ids in iterate_data(len(z), batch_size, shuffle=False):
                x1_gen.append(self.decode(sess, z[batch_ids], **kwargs))

            x1_gen = np.concatenate(x1_gen, axis=0)

        if dec_output_2_img_func is not None:
            x1_gen = dec_output_2_img_func(x1_gen)

        x1_gen = np.reshape(x1_gen, to_list(block_shape) + self.x_shape)
        save_img_block(save_file, x1_gen)

    def reconstruct_images(self, save_file, sess, x, block_shape,
                           show_original_images=True,
                           batch_size=20, dec_output_2_img_func=None, **kwargs):
        if batch_size < 0:
            x1 = self.reconstruct(sess, x, **kwargs)
        else:
            x1 = []
            for batch_ids in iterate_data(len(x), batch_size, shuffle=False):
                x1.append(self.reconstruct(sess, x[batch_ids], **kwargs))

            x1 = np.concatenate(x1, axis=0)

        if dec_output_2_img_func is not None:
            x1 = dec_output_2_img_func(x1)
            x = dec_output_2_img_func(x)

        x1 = np.reshape(x1, to_list(block_shape) + self.x_shape)
        x = np.reshape(x, to_list(block_shape) + self.x_shape)

        if show_original_images:
            save_img_blocks_col_by_col(save_file, [x, x1])
        else:
            save_img_block(save_file, x1)

    def interpolate_images(self, save_file, sess, x1, x2, num_itpl_points,
                           batch_on_row=True, batch_size=20,
                           dec_output_2_img_func=None, enc_kwargs={}, dec_kwargs={}):
        if batch_size < 0:
            z1 = self.encode(sess, x1, **enc_kwargs)
            z2 = self.encode(sess, x2, **enc_kwargs)
        else:
            z1, z2 = [], []
            for batch_ids in iterate_data(len(x1), batch_size, shuffle=False):
                z1.append(self.encode(sess, x1[batch_ids], **enc_kwargs))
                z2.append(self.encode(sess, x2[batch_ids], **enc_kwargs))

            z1 = np.concatenate(z1, axis=0)
            z2 = np.concatenate(z2, axis=0)

        z1_flat = np.ravel(z1)
        z2_flat = np.ravel(z2)

        zs_itpl = []
        for i in range(1, num_itpl_points + 1):
            zi_flat = z1_flat + (i*1.0/(num_itpl_points + 1)) * (z2_flat - z1_flat)
            zs_itpl.append(zi_flat)

        # (num_itpl_points, batch_size * z_dim)
        zs_itpl = np.stack(zs_itpl, axis=0)
        # (num_itpl_points * batch_size, z_shape)
        zs_itpl = np.reshape(zs_itpl, [num_itpl_points * x1.shape[0]] + self.z_shape)

        if batch_size < 0:
            xs_itpl = self.decode(sess, zs_itpl, **dec_kwargs)
        else:
            xs_itpl = []
            for batch_ids in iterate_data(len(zs_itpl), batch_size, shuffle=False):
                xs_itpl.append(self.decode(sess, zs_itpl[batch_ids], **dec_kwargs))

            xs_itpl = np.concatenate(xs_itpl, axis=0)

        # (num_itpl_points, batch_size, x_dim)
        xs_itpl = np.reshape(xs_itpl, [num_itpl_points, x1.shape[0]] + self.x_shape)
        # (num_itpl_points + 2, batch_size, x_dim)
        xs_itpl = np.concatenate([np.expand_dims(x1, axis=0), xs_itpl, np.expand_dims(x2, axis=0)], axis=0)

        if batch_on_row:
            xs_itpl = np.transpose(xs_itpl, [1, 0] + list(range(2, len(self.x_shape) + 2)))

        if dec_output_2_img_func is not None:
            xs_itpl = dec_output_2_img_func(xs_itpl)

        save_img_block(save_file, xs_itpl)

    # These functions are suitable for testing
    # ========================================== #
    # Assume that all z are continuous
    def rand_2_latents_traverse(self, save_file, sess,
                                default_z,
                                z_comp1, start1, stop1, num_points1,
                                z_comp2, start2, stop2, num_points2,
                                batch_size=20,
                                dec_output_2_img_func=None, **kwargs):
        """
        default_z: A single latent code to serve as default
        z_comp1: z component 1
        z_limits1: 2-tuple specifying the low-high value of z_comp1
        num_points1: Number of points
        z_comp2:
        z_limits2:
        num_points2:
        """
        assert num_points1 >= 2, "'num_points1' must be >=2. Found {}!".format(num_points1)
        assert num_points2 >= 2, "'num_points2' must be >=2. Found {}!".format(num_points2)

        z_range1 = [start1 + (stop1 - start1) * i * 1.0 / (num_points1 - 1)
                    for i in range(num_points1)]
        z_range2 = [start2 + (stop2 - start2) * i * 1.0 / (num_points2 - 1)
                    for i in range(num_points2)]

        num_rows = len(z_range1)
        num_cols = len(z_range2)

        assert np.shape(default_z) == tuple(self.z_shape), "'default_z' must be a single instance!"

        default_z = np.reshape(default_z, [int(np.prod(self.z_shape))])
        z_meshgrid = np.tile(np.expand_dims(default_z, axis=0), [num_rows * num_cols, 1])

        for m in range(num_rows):
            for n in range(num_cols):
                z_meshgrid[m * num_cols + n, z_comp1] = z_range1[m]
                z_meshgrid[m * num_cols + n, z_comp2] = z_range2[n]

        # Reconstruct x meshgrid
        # ----------------------------- #
        if batch_size < 0:
            x_meshgrid = self.decode(sess, z_meshgrid, **kwargs)
        else:
            x_meshgrid = []
            for batch_ids in iterate_data(len(z_meshgrid), batch_size, shuffle=False):
                x_meshgrid.append(self.decode(sess, z_meshgrid[batch_ids], **kwargs))

            x_meshgrid = np.concatenate(x_meshgrid, axis=0)

        x_meshgrid = np.reshape(x_meshgrid, [num_rows, num_cols] + self.x_shape)

        if dec_output_2_img_func is not None:
            x_meshgrid = dec_output_2_img_func(x_meshgrid)

        save_img_block(save_file, x_meshgrid)
        # ----------------------------- #

    # Assume that all z are continuous
    # x is a single image
    def cond_2_latents_traverse(self, save_file, sess, x,
                                z_comp1, span1, num_points_one_side1,
                                z_comp2, span2, num_points_one_side2,
                                batch_size=20,
                                hl_color="red", hl_width=1,
                                dec_output_2_img_func=None,
                                enc_kwargs={}, dec_kwargs={}):
        """
        x: A SINGLE input image that we condition on
        z_comp1: An integer, specifying which z component we want to plot
        z_span1: The distance from the center value of z1 when we encode x
        z_comp2: An integer, specifying the other z component we want to plot
        z_span2: The distance from the center value of z2 when we encode x
        num_itpl_points_4_one_side: We have 2 sides and a conditional input x in the middle.
        This value describe the number of interpolation points we want for each side
        """
        assert np.shape(x) == tuple(self.x_shape), "'x' must be a single instance!"
        # (1, x_dim)
        x_ = np.expand_dims(x, axis=0)

        # Compute z
        # ----------------------------- #
        # (1, z_dim)
        z = self.encode(sess, x_, **enc_kwargs)
        assert z.shape[0] == 1

        # (z_dim, )
        z = np.reshape(z, [int(np.prod(self.z_shape))])
        # ----------------------------- #

        # Compute z meshgrid
        # ----------------------------- #
        # Compute 'z_range1' and 'z_range2'
        # (num_rows * num_cols, z_dim)
        z12_meshgrid, center_idx = get_meshgrid((z[z_comp1], z[z_comp2]), (span1, span2),
                                                (num_points_one_side1, num_points_one_side2),
                                                return_center_idx=True)

        num_rows = 2 * num_points_one_side1 + 1
        num_cols = 2 * num_points_one_side2 + 1
        assert len(z12_meshgrid) == num_rows * num_cols

        z_meshgrid = np.tile(np.expand_dims(z, axis=0), [num_rows * num_cols, 1])
        for i in range(num_rows * num_cols):
            z_meshgrid[i, z_comp1] = z12_meshgrid[i, 0]
            z_meshgrid[i, z_comp2] = z12_meshgrid[i, 1]

        z_meshgrid = np.reshape(z_meshgrid, [num_rows * num_cols] + self.z_shape)
        # ----------------------------- #

        # Reconstruct x meshgrid
        # ----------------------------- #
        if batch_size < 0:
            x_meshgrid = self.decode(sess, z_meshgrid, **dec_kwargs)
        else:
            x_meshgrid = []
            for batch_ids in iterate_data(len(z_meshgrid), batch_size, shuffle=False):
                x_meshgrid.append(self.decode(sess, z_meshgrid[batch_ids], **dec_kwargs))

            x_meshgrid = np.concatenate(x_meshgrid, axis=0)

        x_meshgrid[center_idx] = x

        x_meshgrid = np.reshape(x_meshgrid, [num_rows, num_cols] + self.x_shape)

        if dec_output_2_img_func is not None:
            x_meshgrid = dec_output_2_img_func(x_meshgrid)

        center_block_idx = (center_idx/num_cols, center_idx % num_cols)
        save_img_block_highlighted(save_file, x_meshgrid, [center_block_idx],
                                   hl_color=hl_color, hl_width=hl_width)
        # ----------------------------- #

    def cond_all_latents_traverse(self, save_file, sess, x,
                                  z_comps=None, z_comp_labels=None,
                                  start=-3.0, stop=3.0, num_itpl_points=10,
                                  hl_color="red", hl_width=1, subplot_adjust={},
                                  batch_size=20,
                                  dec_output_2_img_func=None,
                                  enc_kwargs={}, dec_kwargs={}):

        assert num_itpl_points >= 2, "'num_points' must be >= 2!"
        itpl_points = [start + (stop - start) * i * 1.0 / (num_itpl_points - 1)
                       for i in range(0, num_itpl_points)]

        assert np.shape(x) == tuple(self.x_shape), "'x' must be a single instance!"
        # (1, x_dim)
        x_ = np.expand_dims(x, axis=0)

        # Compute z
        # ----------------------------- #
        # (1, z_dim)
        z = self.encode(sess, x_, **enc_kwargs)
        assert z.shape[0] == 1

        # (z_dim, )
        z_dim = int(np.prod(self.z_shape))
        z = np.reshape(z, [z_dim])
        # ----------------------------- #

        if z_comps is None:
            z_comps = list(range(z_dim))

        z_meshgrid = []
        inserted_ids = []

        for i, comp in enumerate(z_comps):
            itpl_zi = []
            idx = 0

            for k in range(len(itpl_points)):
                z_copy = np.array(z, dtype=z.dtype, copy=True)
                z_copy[comp] = itpl_points[k]
                itpl_zi.append(z_copy)

                if (1 <= k) and itpl_points[k-1] <= z[comp] < itpl_points[k]:
                    idx = k

            if itpl_points[len(itpl_points)-1] <= z[comp]:
                idx = len(itpl_points)

            inserted_ids.append((i, idx))

            itpl_zi.insert(idx, np.array(z, dtype=z.dtype, copy=True))
            z_meshgrid.extend(itpl_zi)

        # Compute z meshgrid
        # ----------------------------- #
        num_rows = len(z_comps)
        num_cols = num_itpl_points + 1
        assert len(z_meshgrid) == num_rows * num_cols

        z_meshgrid = np.reshape(z_meshgrid, [num_rows * num_cols] + self.z_shape)
        # ----------------------------- #

        # Reconstruct x meshgrid
        # ----------------------------- #
        if batch_size < 0:
            x_meshgrid = self.decode(sess, z_meshgrid, **dec_kwargs)
        else:
            x_meshgrid = []
            for batch_ids in iterate_data(len(z_meshgrid), batch_size, shuffle=False):
                x_meshgrid.append(self.decode(sess, z_meshgrid[batch_ids], **dec_kwargs))

            x_meshgrid = np.concatenate(x_meshgrid, axis=0)

        x_meshgrid = np.reshape(x_meshgrid, [num_rows, num_cols] + self.x_shape)
        for row_idx, col_idx in inserted_ids:
            x_meshgrid[row_idx, col_idx] = x

        if dec_output_2_img_func is not None:
            x_meshgrid = dec_output_2_img_func(x_meshgrid)

        if z_comp_labels is not None:
            assert len(z_comp_labels) == len(z_comps), \
                "Length of 'z_comp_labels' must be equal to the number of z components " \
                "you want to draw. Found {} and {}, respectively!".format(len(z_comp_labels), len(z_comps))
            save_img_block_highlighted_with_ticklabels(
                save_file, x_meshgrid,
                hl_blocks=inserted_ids, hl_color=hl_color, hl_width=hl_width,
                x_tick_labels=None,
                y_tick_labels=z_comp_labels, subplot_adjust=subplot_adjust)
        else:
            save_img_block_highlighted(save_file, x_meshgrid, hl_blocks=inserted_ids,
                                       hl_color=hl_color, hl_width=hl_width)
        # ----------------------------- #

    def cond_all_latents_traverse_v2(self, save_file, sess, x,
                                     z_comps=None, z_comp_labels=None,
                                     span=2, points_1_side=6,
                                     # substitute with original x and highlight
                                     hl_x=True, hl_color="red", hl_width=1,
                                     font_size=12, title="", title_font_scale=1.5,
                                     subplot_adjust={}, size_inches=None,
                                     batch_size=20, dec_output_2_img_func=None,
                                     enc_kwargs={}, dec_kwargs={}):

        assert np.shape(x) == tuple(self.x_shape), "'x' must be a single instance!"
        # (1, x_dim)
        x_ = np.expand_dims(x, axis=0)

        # Compute z
        # ----------------------------- #
        # (1, z_dim)
        z = self.encode(sess, x_, **enc_kwargs)
        assert z.shape[0] == 1

        # (z_dim, )
        z_dim = int(np.prod(self.z_shape))
        z = np.reshape(z, [z_dim])
        # ----------------------------- #

        if z_comps is None:
            z_comps = list(range(z_dim))

        z_meshgrid = []
        inserted_ids = []
        s = span
        p = points_1_side

        for i, comp in enumerate(z_comps):
            # (2 * points_1_side + 1, )
            itpl_vals = [(z[comp] - s) + 1.0 * i * s / p for i in range(p)]
            itpl_vals += [z[comp]]
            itpl_vals += [z[comp] + 1.0 * i * s / p for i in range(1, p + 1)]

            for val in itpl_vals:
                z_copy = np.array(z, dtype=z.dtype, copy=True)
                z_copy[comp] = val
                z_meshgrid.append(z_copy)

            inserted_ids.append((i, points_1_side))

        # Compute z meshgrid
        # ----------------------------- #
        num_rows = len(z_comps)
        num_cols = 2 * points_1_side + 1
        assert len(z_meshgrid) == num_rows * num_cols

        z_meshgrid = np.reshape(z_meshgrid, [num_rows * num_cols] + self.z_shape)
        # ----------------------------- #

        # Reconstruct x meshgrid
        # ----------------------------- #
        if batch_size < 0:
            x_meshgrid = self.decode(sess, z_meshgrid, **dec_kwargs)
        else:
            x_meshgrid = []
            for batch_ids in iterate_data(len(z_meshgrid), batch_size, shuffle=False):
                x_meshgrid.append(self.decode(sess, z_meshgrid[batch_ids], **dec_kwargs))

            x_meshgrid = np.concatenate(x_meshgrid, axis=0)

        x_meshgrid = np.reshape(x_meshgrid, [num_rows, num_cols] + self.x_shape)

        if hl_x:
            for row_idx, col_idx in inserted_ids:
                x_meshgrid[row_idx, col_idx] = x

        if dec_output_2_img_func is not None:
            x_meshgrid = dec_output_2_img_func(x_meshgrid)

        if z_comp_labels is not None:
            assert len(z_comp_labels) == len(z_comps), \
                "Length of 'z_comp_labels' must be equal to the number of z components " \
                "you want to draw. Found {} and {}, respectively!".format(len(z_comp_labels), len(z_comps))

            if hl_x:
                save_img_block_highlighted_with_ticklabels(
                    save_file, x_meshgrid,
                    hl_blocks=inserted_ids, hl_color=hl_color, hl_width=hl_width,
                    x_tick_labels=None, y_tick_labels=z_comp_labels,
                    font_size=font_size, title=title, title_font_scale=title_font_scale,
                    subplot_adjust=subplot_adjust, size_inches=size_inches)
            else:
                save_img_block_with_ticklabels(
                    save_file, x_meshgrid,
                    x_tick_labels=None, y_tick_labels=z_comp_labels,
                    font_size=font_size, title=title, title_font_scale=title_font_scale,
                    subplot_adjust=subplot_adjust, size_inches=size_inches)
        else:
            if hl_x:
                save_img_block_highlighted(save_file, x_meshgrid, hl_blocks=inserted_ids,
                                           hl_color=hl_color, hl_width=hl_width)
            else:
                save_img_block(save_file, x_meshgrid)
        # ----------------------------- #

    def cond_1_latent_traverse(self, save_file, sess, x, z_comp,
                               start=-3.0, stop=3.0, num_itpl_points=10,
                               x_labels=None,
                               hl_color="red", hl_width=1, subplot_adjust={},
                               batch_size=20,
                               dec_output_2_img_func=None,
                               enc_kwargs={}, dec_kwargs={}):

        assert num_itpl_points >= 2, "'num_points' must be >= 2!"
        itpl_points = [start + (stop - start) * i * 1.0 / (num_itpl_points - 1)
                       for i in range(0, num_itpl_points)]

        assert (len(x.shape) == len(self.x_shape) + 1) and (x.shape[1:] == tuple(self.x_shape)), \
            "'x' must contain batch dimension. Found x.shape={}!".format(x.shape)

        # Compute z
        # ----------------------------- #
        # (batch, z_shape)
        z = self.encode(sess, x, **enc_kwargs)

        # (batch, z_dim)
        z_dim = int(np.prod(self.z_shape))
        z = np.reshape(z, [x.shape[0], z_dim])
        # ----------------------------- #

        z_meshgrid = []
        inserted_ids = []

        for n in range(x.shape[0]):
            itpl_zn = []
            idx = 0

            for k in range(len(itpl_points)):
                z_copy = np.array(z[n], dtype=z.dtype, copy=True)
                z_copy[z_comp] = itpl_points[k]
                itpl_zn.append(z_copy)

                if (1 <= k) and itpl_points[k-1] <= z[n, z_comp] < itpl_points[k]:
                    idx = k

            if itpl_points[len(itpl_points)-1] <= z[n, z_comp]:
                idx = len(itpl_points)

            inserted_ids.append((n, idx))

            itpl_zn.insert(idx, np.array(z[n], dtype=z.dtype, copy=True))
            z_meshgrid.extend(itpl_zn)

        # Compute z meshgrid
        # ----------------------------- #
        num_rows = x.shape[0]
        num_cols = num_itpl_points + 1
        assert len(z_meshgrid) == num_rows * num_cols

        z_meshgrid = np.reshape(z_meshgrid, [num_rows * num_cols] + self.z_shape)
        # ----------------------------- #

        # Reconstruct x meshgrid
        # ----------------------------- #
        if batch_size < 0:
            x_meshgrid = self.decode(sess, z_meshgrid, **dec_kwargs)
        else:
            x_meshgrid = []
            for batch_ids in iterate_data(len(z_meshgrid), batch_size, shuffle=False):
                x_meshgrid.append(self.decode(sess, z_meshgrid[batch_ids], **dec_kwargs))

            x_meshgrid = np.concatenate(x_meshgrid, axis=0)

        x_meshgrid = np.reshape(x_meshgrid, [num_rows, num_cols] + self.x_shape)
        for row_idx, col_idx in inserted_ids:
            x_meshgrid[row_idx, col_idx] = x[row_idx]

        if dec_output_2_img_func is not None:
            x_meshgrid = dec_output_2_img_func(x_meshgrid)

        if x_labels is not None:
            save_img_block_highlighted_with_ticklabels(
                save_file, x_meshgrid, hl_blocks=inserted_ids,
                hl_color=hl_color, hl_width=hl_width,
                x_tick_labels=None,
                y_tick_labels=x_labels, subplot_adjust=subplot_adjust)
        else:
            save_img_block_highlighted(
                save_file, x_meshgrid, hl_blocks=inserted_ids,
                hl_color=hl_color, hl_width=hl_width)
        # ----------------------------- #
    # ========================================== #

    def plot_Z_itpl_1X(self, save_file_prefix, sess, imgs, img_names,
                       features, start=-3.0, stop=3.0, num_itpl_points=10,
                       yx_types=('feature', 'itpl_point'),
                       dec_output_2_img_func=None, img_ext='png', batch_size=-1):

        # img_ext
        # ---------------------------------------- #
        assert img_ext == 'png' or img_ext == 'jpg', "'img_ext' must be png or jpg!"
        # ---------------------------------------- #

        # coordinate
        # ---------------------------------------- #
        # For this kind of interpolation, the results will have 3 axes:
        # (num_inputs, num_features, num_itpl_points)

        # If we set mode == 'share_inputs', we will have 'num_inputs' block images
        # of shape (num_features, num_itpl_points)
        possible_coord_types = [('input', 'itpl_point'), ('feature', 'itpl_point'),
                                ('itpl_point', 'input'), ('itpl_point', 'feature')]

        if isinstance(yx_types, tuple):
            assert len(yx_types) == 2, "'yx_types' must be a 2-tuples or " \
                                       "a list of 2-tuples representing the yx coordinate types!"
            yx_types = [yx_types]

        assert isinstance(yx_types, list), "'yx_types' must be a 2-tuples or " \
                                           "a list of 2-tuples representing the yx coordinate types!"

        assert all([yx_type in possible_coord_types for yx_type in yx_types]), \
            "Only support the following coordinate types: {}".format(possible_coord_types)
        # ---------------------------------------- #

        # num_images
        # ---------------------------------------- #
        assert isinstance(imgs, np.ndarray) and imgs.ndim == 4, \
            "'imgs' must be a 4D numpy array of format (num_images, height, width, channels)!"
        num_inputs = imgs.shape[0]
        # ---------------------------------------- #

        # num_features
        # ---------------------------------------- #
        z_dim = int(np.prod(self.z_shape))
        if features == 'all':
            features = [i for i in range(z_dim)]

        if isinstance(features, int):
            assert 0 <= features < z_dim, "'features' must be an integer or " \
                "a list/tuple of integers in the range [0, {}]".format(z_dim - 1)
            features = [features]

        assert isinstance(features, (list, tuple)), "'features' must be an integer or " \
            "a list/tuple of integers in the range [0, {}]".format(z_dim - 1)

        num_features = len(features)
        # ---------------------------------------- #

        # num_itpl_points
        # ---------------------------------------- #
        assert num_itpl_points >= 2, "'num_points' must be >= 2!"
        itpl_points = [start + (stop - start) * i * 1.0 / (num_itpl_points - 1)
                       for i in range(0, num_itpl_points)]
        # ---------------------------------------- #

        # (num_images, z_dim)
        z = np.reshape(self.encode(sess, imgs), [num_inputs, z_dim])

        z_samples = []  # (num_features * num_itpl_points) of (num_images, z_dim) array

        for feature in features:
            for itpl_point in itpl_points:
                z_copy = np.array(z, dtype=z.dtype, copy=True)
                z_copy[:, feature] = itpl_point

                z_samples.append(z_copy)

        # (num_features * num_itpl_points * num_images, z_dim)
        z_samples = np.concatenate(z_samples, axis=0)

        if batch_size < 0:
            z_samples = np.reshape(z_samples, [num_inputs * num_features * num_itpl_points] + self.z_shape)
            x_samples = self.decode(sess, z_samples)
        else:
            x_samples = []
            for batch_ids in iterate_data(len(z_samples), batch_size, shuffle=False):
                x_samples.append(self.decode(sess, np.reshape(z_samples[batch_ids], [len(batch_ids)] + self.z_shape)))
            x_samples = np.concatenate(x_samples, axis=0)

        # (num_features, num_itpl_points, num_images) + x_shape
        x_samples = np.reshape(x_samples, [num_features, num_itpl_points, num_inputs] + self.x_shape)
        if dec_output_2_img_func is not None:
            x_samples = dec_output_2_img_func(x_samples)

        for yx_type in yx_types:
            if yx_type == ('itpl_point', 'input'):
                x_itpl = x_samples
                save_file_postfixes = ["-feat[{}].{}".format(feature, img_ext) for feature in features]

            elif yx_type == ('input', 'itpl_point'):
                x_itpl = np.transpose(x_samples, [0, 2, 1] + list(range(3, 3 + len(self.x_shape))))
                save_file_postfixes = ["-feat[{}].{}".format(feature, img_ext) for feature in features]

            elif yx_type == ('itpl_point', 'feature'):
                x_itpl = np.transpose(x_samples, [2, 1, 0] + list(range(3, 3 + len(self.x_shape))))
                assert img_names, "'inp_img_names' must be provided!"
                save_file_postfixes = ["-img[{}].{}".format(img_names[i], img_ext) for i in range(len(x_itpl))]

            elif yx_type == ('feature', 'itpl_point'):
                x_itpl = np.transpose(x_samples, [2, 0, 1] + list(range(3, 3 + len(self.x_shape))))
                assert img_names, "'inp_img_names' must be provided!"
                save_file_postfixes = ["-img[{}].{}".format(img_names[i], img_ext) for i in range(len(x_itpl))]

            elif yx_type == (None, 'itpl_point'):
                # (num_images, num_features, num_itpl_points) + x_shape
                x_itpl = np.transpose(x_samples, [2, 0, 1] + list(range(3, 3 + len(self.x_shape))))
                x_itpl = np.reshape(x_itpl, [num_inputs * num_features, 1, num_itpl_points] + self.x_shape)
                save_file_postfixes = ["-img[{}]_feat[{}].{}".format(img_name, feature, img_ext)
                                       for img_name in img_names for feature in features]

            elif yx_type == ('itpl_point', None):
                # (num_images, num_features, num_itpl_points) + x_shape
                x_itpl = np.transpose(x_samples, [2, 0, 1] + list(range(3, 3 + len(self.x_shape))))
                x_itpl = np.reshape(x_itpl, [num_inputs * num_features, num_itpl_points, 1] + self.x_shape)
                save_file_postfixes = ["-img[{}]_feat[{}].{}".format(img_name, feature, img_ext)
                                       for img_name in img_names for feature in features]

            else:
                raise ValueError("Only support the following coordinate types: {}!".format(possible_coord_types))

            for i in range(len(x_itpl)):
                save_file = save_file_prefix + save_file_postfixes[i]
                save_img_block(save_file, x_itpl[i])
    
    # We want to interpolate 1 feature of img1 to reach the value of img2
    # while other features (of img1) are fixed
    def plot_Z_itpl_bw_2Xs(self, save_file_prefix, sess, imgs_1, imgs_2,
                           img_names_1, img_names_2,
                           features, num_itpl_points=6,
                           yx_types=('feature', 'itpl_point'),
                           dec_output_2_img_func=None, img_ext='png', batch_size=-1):

        # img_ext
        # ---------------------------------------- #
        assert img_ext == 'png' or img_ext == 'jpg', "'img_ext' must be png or jpg!"
        # ---------------------------------------- #

        # coordinate
        # ---------------------------------------- #
        # For this kind of interpolation, the results will have 3 axes:
        # (num_inputs, num_features, num_itpl_points)

        # If we set mode == 'share_inputs', we will have 'num_inputs' block images
        # of shape (num_features, num_itpl_points)
        possible_coord_types = [('input', 'itpl_point'), ('feature', 'itpl_point'),
                                ('itpl_point', 'input'), ('itpl_point', 'feature')]

        if isinstance(yx_types, tuple):
            assert len(yx_types) == 2, "'yx_types' must be a 2-tuples or " \
                                       "a list of 2-tuples representing the yx coordinate types!"
            yx_types = [yx_types]

        assert isinstance(yx_types, list), "'yx_types' must be a 2-tuples or " \
                                           "a list of 2-tuples representing the yx coordinate types!"

        assert all([yx_type in possible_coord_types for yx_type in yx_types]), \
            "Only support the following coordinate types: {}".format(possible_coord_types)
        # ---------------------------------------- #

        # num_images
        # ---------------------------------------- #
        assert isinstance(imgs_1, np.ndarray) and imgs_1.ndim == 4, \
            "'inp_imgs_1' must be a 4D numpy array of format (num_images, height, width, channels)!"

        assert isinstance(imgs_2, np.ndarray) and imgs_2.ndim == 4, \
            "'inp_imgs_2' must be None or a 4D numpy array of format (num_images, height, width, channels)!"

        assert len(imgs_1) == len(imgs_2), "Number of images in 'inp_imgs_1' and 'inp_imgs_2' must be equal!"
        num_inputs = len(imgs_1)
        # ---------------------------------------- #

        # num_features
        # ---------------------------------------- #
        z_dim = int(np.prod(self.z_shape))
        if features == 'all':
            features = [i for i in range(z_dim)]

        if isinstance(features, int):
            assert 0 <= features < z_dim, "'features' must be an integer or " \
                "a list/tuple of integers in the range [0, {}]".format(z_dim - 1)
            features = [features]

        assert isinstance(features, (list, tuple)), "'features' must be an integer or " \
            "a list/tuple of integers in the range [0, {}]".format(z_dim - 1)

        num_features = len(features)
        # ---------------------------------------- #

        # (num_images, z_dim)
        z1 = np.reshape(self.encode(sess, imgs_1), [num_inputs, z_dim])
        z2 = np.reshape(self.encode(sess, imgs_2), [num_inputs, z_dim])

        z_samples = []  # (num_features * num_itpl_points) of (num_images, z_dim) array

        for n in range(len(imgs_1)):
            for feature in features:
                # (num_itpl_points, )
                itpl_points = np.linspace(z1[n, feature], z2[n, feature], num=num_itpl_points, endpoint=True)

                for itpl_point in itpl_points:
                    z_copy = np.array(z1[n], dtype=z1.dtype, copy=True)
                    z_copy[feature] = itpl_point

                    z_samples.append(z_copy)

        # (num_inputs * num_features * num_itpl_points, z_dim)
        z_samples = np.stack(z_samples, axis=0)

        if batch_size < 0:
            z_samples = np.reshape(z_samples, [num_inputs * num_features * num_itpl_points] + self.z_shape)
            x_samples = self.decode(sess, z_samples)
        else:
            x_samples = []
            for batch_ids in iterate_data(len(z_samples), batch_size, shuffle=False):
                x_samples.append(self.decode(sess, np.reshape(z_samples[batch_ids],
                                                              [len(batch_ids)] + self.z_shape)))
            x_samples = np.concatenate(x_samples, axis=0)

        # (num_images, num_features, num_itpl_points) + x_shape
        x_samples = np.reshape(x_samples, [num_inputs, num_features, num_itpl_points] + self.x_shape)
        if dec_output_2_img_func is not None:
            x_samples = dec_output_2_img_func(x_samples)

        for yx_type in yx_types:
            if yx_type == ('feature', 'itpl_point'):
                x_itpl = x_samples
                assert img_names_1, "'inp_img_names_1' must be provided!"
                assert img_names_2, "'inp_img_names_2' must be provided!"
                save_file_postfixes = ["-img[{}-{}].{}".format(img_names_1[i], img_names_2[i], img_ext)
                                       for i in range(len(x_itpl))]

            elif yx_type == ('itpl_point', 'feature'):
                x_itpl = np.transpose(x_samples, [0, 2, 1] + list(range(3, 3 + len(self.x_shape))))
                assert img_names_1, "'inp_img_names_1' must be provided!"
                assert img_names_2, "'inp_img_names_2' must be provided!"
                save_file_postfixes = ["-img[{}-{}].{}".format(img_names_1[i], img_names_2[i], img_ext)
                                       for i in range(len(x_itpl))]

            elif yx_type == ('input', 'itpl_point'):
                x_itpl = np.transpose(x_samples, [1, 0, 2] + list(range(3, 3 + len(self.x_shape))))
                save_file_postfixes = ["-feat[{}].{}".format(feature, img_ext) for feature in features]

            elif yx_type == ('itpl_point', 'input'):
                x_itpl = np.transpose(x_samples, [1, 2, 0] + list(range(3, 3 + len(self.x_shape))))
                save_file_postfixes = ["-feat[{}].{}".format(feature, img_ext) for feature in features]

            elif yx_type == (None, 'itpl_point'):
                # (num_images, num_features, num_itpl_points) + x_shape
                x_itpl = np.reshape(x_samples, [num_inputs * num_features, 1, num_itpl_points] + self.x_shape)
                save_file_postfixes = ["-img[{}-{}]_feat[{}].{}".format(img_name_1, img_name_2, feature, img_ext)
                                       for img_name_1, img_name_2 in zip(img_names_1, img_name_2)
                                       for feature in features]

            elif yx_type == ('itpl_point', None):
                # (num_images, num_features, num_itpl_points) + x_shape
                x_itpl = np.reshape(x_samples, [num_inputs * num_features, num_itpl_points, 1] + self.x_shape)
                save_file_postfixes = ["-img[{}-{}]_feat[{}].{}".format(img_name_1, img_name_2, feature, img_ext)
                                       for img_name_1, img_name_2 in zip(img_names_1, img_name_2)
                                       for feature in features]

            else:
                raise ValueError("Only support the following coordinate types: {}".format(possible_coord_types))

            for i in range(len(x_itpl)):
                save_file = save_file_prefix + save_file_postfixes[i]
                save_img_block(save_file, x_itpl[i])

    @staticmethod
    def gp(center, mode, disc_fn, inp_1, inp_2, at_D_comp=None):
        if mode == "interpolation":
            print("[BaseLatentModel.gp] interpolation!")
            assert (inp_1 is not None) and (inp_2 is not None), "If 'mode' is 'interpolation', " \
                "both 'inp_1' and 'inp_2' must be provided!"

            # IMPORTANT: The formula below is wrong (compared to the original).
            # IMPORTANT: It must be [batch, 1, 1, ...], not mixed_shape(inp_1)
            # alpha = tf.random_uniform(shape=mixed_shape(inp_1), minval=0., maxval=1.)

            shape = mixed_shape(inp_1)
            batch_size = shape[0]
            # (batch, 1)
            alpha = tf.random_uniform(shape=[batch_size] + [1] * (len(shape) - 1), minval=0., maxval=1.)

            # (batch, z_dim)
            inp_i = alpha * inp_1 + ((1 - alpha) * inp_2)

            if at_D_comp is not None:
                D_logit = disc_fn(inp_i)
                D_logit_i = D_logit[:, at_D_comp]
            else:
                D_logit_i = disc_fn(inp_i)

            assert len(D_logit_i.shape.as_list()) == 1, D_logit_i.shape.as_list()

            D_grad = tf.gradients(D_logit_i, [inp_i])[0]
            D_grad = flatten_right_from(D_grad, 1)

        elif mode == "self":
            print("[BaseLatentModel.gp] self!")
            assert inp_1 is not None, "If 'mode' is 'self', 'inp_1' must be provided!"

            if at_D_comp is not None:
                D_logit = disc_fn(inp_1)
                D_logit_1 = D_logit[:, at_D_comp]
            else:
                D_logit_1 = disc_fn(inp_1)

            assert len(D_logit_1.shape.as_list()) == 1, D_logit_1.shape.as_list()

            D_grad = tf.gradients(D_logit_1, [inp_1])[0]
            D_grad = flatten_right_from(D_grad, 1)

        elif mode == "other":
            print("[BaseLatentModel.gp] other!")
            assert inp_2 is not None, "If 'mode' is 'other', 'inp_2' must be provided!"

            if at_D_comp is not None:
                D_logit = disc_fn(inp_2)
                D_logit_2 = D_logit[:, at_D_comp]
            else:
                D_logit_2 = disc_fn(inp_2)

            assert len(D_logit_2.shape.as_list()) == 1, D_logit_2.shape.as_list()

            D_grad = tf.gradients(D_logit_2, [inp_2])[0]
            D_grad = flatten_right_from(D_grad, 1)
            assert len(D_grad.shape) == 2, "'D_grad' must have 2 dimensions!"

        else:
            raise ValueError("Do not support 'mode'='{}'".format(mode))

        assert len(D_grad.shape) == 2, "'D_grad' must have 2 dimensions!"

        if center == 0:
            gp = tf.reduce_mean(tf.reduce_sum(tf.square(D_grad), axis=1), axis=0)
        else:
            slope = tf.sqrt(tf.reduce_sum(tf.square(D_grad), axis=1))
            gp = tf.reduce_mean((slope - center) ** 2, axis=0)

        return gp

    @staticmethod
    def gp0(mode, disc_fn, inp_1, inp_2, at_D_comp=None):
        return BaseLatentModel.gp(0, mode, disc_fn, inp_1, inp_2, at_D_comp)

    @staticmethod
    def gp_4_inp_list(center, mode, disc_fn, inputs_1, inputs_2, at_D_comp=None):
        if mode == "interpolation":
            print("[BaseLatentModel.gp_4_inp_list] interpolation!")
            assert isinstance(inputs_1, (list, tuple))
            assert isinstance(inputs_1, (list, tuple))

            inputs_i = []

            for inp_1, inp_2 in zip(inputs_1, inputs_2):
                alpha = tf.random_uniform(shape=mixed_shape(inp_1), minval=0., maxval=1.)
                # (batch, z_dim)
                inp_i = alpha * inp_1 + ((1 - alpha) * inp_2)
                inputs_i.append(inp_i)

            if at_D_comp is not None:
                D_logit_i = disc_fn(inputs_i)
                D_logit_i = D_logit_i[:, at_D_comp]
            else:
                D_logit_i = disc_fn(inputs_i)

            assert len(D_logit_i.shape.as_list()) == 1, D_logit_i.shape.as_list()

            # List of gradient for each input component
            D_grads = tf.gradients(D_logit_i, inputs_i)
            D_grads = [flatten_right_from(Dg, axis=1) for Dg in D_grads]

        else:
            raise ValueError("Do not support 'mode'='{}'".format(mode))

        gp = tf.constant(0, dtype=tf.float32)

        for Dg in D_grads:
            assert len(Dg.shape) == 2, "'Dg' must have 2 dimensions!"
            slope = tf.sqrt(tf.reduce_sum(tf.square(Dg), axis=1))
            gp += tf.reduce_mean((slope - center) ** 2, axis=0)

        return gp

    @staticmethod
    def gp0_4_inp_list(mode, disc_fn, inp_1, inp_2, at_D_comp=None):
        return BaseLatentModel.gp_4_inp_list(0, mode, disc_fn, inp_1, inp_2, at_D_comp)

    @staticmethod
    def gp0_indv(mode, disc_indv_fn, inp_1, inp_2):
        # Gradient penalty for individual discriminator
        if mode == "interpolation":
            print("[BaseLatentModel.gp0_indv] interpolation!")
            assert (inp_1 is not None) and (inp_2 is not None), "If 'mode' is 'interpolation', " \
                                                                "both 'inp_1' and 'inp_2' must be provided!"

            alpha = tf.random_uniform(shape=mixed_shape(inp_1), minval=0., maxval=1.)
            # (batch, z_dim)
            inp_i = alpha * inp_1 + ((1 - alpha) * inp_2)
            # (batch, z_dim)
            D_logit_i = disc_indv_fn(inp_i)

            D_grad_i = tf.gradients(D_logit_i, [inp_i])[0]
            D_grad_i = flatten_right_from(D_grad_i, 1)
            assert len(D_grad_i.shape) == 2, "'D_grad_i' must have 2 dimensions!"

            gp = tf.reduce_mean(tf.reduce_sum(tf.square(D_grad_i), axis=1), axis=0)

        elif mode == "self":
            print("[BaseLatentModel.gp0_indv] self!")
            assert inp_1 is not None, "If 'mode' is 'self', 'inp_1' must be provided!"

            D_logit_1 = disc_indv_fn(inp_1)

            D_grad_1 = tf.gradients(D_logit_1, [inp_1])[0]
            D_grad_1 = flatten_right_from(D_grad_1, 1)
            assert len(D_grad_1.shape) == 2, "'D_grad_1' must have 2 dimensions!"

            gp = tf.reduce_mean(tf.reduce_sum(tf.square(D_grad_1), axis=1), axis=0)

        elif mode == "other":
            print("[BaseLatentModel.gp0_indv] other!")
            assert inp_2 is not None, "If 'mode' is 'other', 'inp_2' must be provided!"

            D_logit_2 = disc_indv_fn(inp_2)

            D_grad_2 = tf.gradients(D_logit_2, [inp_2])[0]
            D_grad_2 = flatten_right_from(D_grad_2, 1)
            assert len(D_grad_2.shape) == 2, "'D_grad_2' must have 2 dimensions!"

            gp = tf.reduce_mean(tf.reduce_sum(tf.square(D_grad_2), axis=1), axis=0)

        else:
            raise ValueError("Do not support 'mode'='{}'".format(mode))

        return gp