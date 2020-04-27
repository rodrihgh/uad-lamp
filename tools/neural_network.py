import os
import numpy as np
from numpy.linalg import inv
import tensorflow as tf
from tensorflow.image import hsv_to_rgb, resize_nearest_neighbor


def pseudoinverse(a, reg=True):
    """
    Regularized pseudo-inverse for a complex matrix
    @param a: Matrix to invert
    @param reg: whether to regularize for tr(AB_0)=N or not
    @return: (regularized) Complex pseudo-inverse
    """

    n_dims = a.ndim
    dim1 = n_dims - 2
    dim2 = n_dims - 1
    m = a.shape[dim1]
    n = a.shape[dim2]

    # Permute right axes
    axes = [axis for axis in range(a.ndim)]
    axes[dim1] = dim2
    axes[dim2] = dim1

    a_h = np.conj(np.transpose(a, axes))
    b = np.matmul(a_h, inv(np.matmul(a, a_h) + np.eye(m)))

    if reg:
        gamma = np.trace(np.matmul(a, b), axis1=dim1, axis2=dim2) / n
        gamma = np.expand_dims(gamma, -1)
        gamma = np.expand_dims(gamma, -1)
        b /= gamma

    return b


class LAMP:

    (TF_REAL, TF_COMPLEX) = (tf.float64, tf.complex128)

    def __init__(self, x, y, n, m, a, b, theta, batch_size, num_layers=1,
                 b_train=True, theta_train=True, nmse=False, shrink_func='scaled_soft_th',
                 tied=True, lr=0.001, save_results=True, analyze_data=True,
                 squared=True, save_support=True, train=True,
                 log_dir=os.getcwd(), seed=0):
        """

        @param x: training data for the reconstructed signals
        @param y: training data for the measured signals
        @param n: size of reconstructed signals
        @param m: size of measured signals
        @param a: measuring matrix A
        @param b: initial values for B, the learnable inverse of A
        @param theta: initial values for theta
        @param batch_size: batch size for training
        @param num_layers: total number of AMP layers to unfold
        @param b_train: list indicating which Bs should be trained. If a boolean, determines the train status \
        of all parameters across layers
        @type b_train: Union[bool, list]
        @param theta_train: list indicating which thetas should be trained. If a boolean, determines the train status \
        of all parameters across layers
        @type theta_train: Union[bool, list]
        @param tied: whether all layers should share the same B
        @param lr: learning rate
        @param save_results: whether to save training statistics
        @param analyze_data: whether to save input / output statistics
        @param squared: whether to use L2 (default) or L1
        @param save_support: whether to save support comparison of x and y
        @param log_dir: directory to save training statistics
        @param seed: random seed for tensorflow
        """

        tf.reset_default_graph()

        self.summary = None
        self.writer = None
        self.session = None

        self.tied = tied
        self.save_results = save_results
        self.analyze_data = analyze_data
        self.log_dir = log_dir
        self.learning_rate = lr

        self.batcher = self.Batcher(batch_size, x.shape[0])
        self.T = num_layers

        # Assign constant tensors
        self.N = n
        self.M = m

        if a.dtype == self.TF_REAL.as_numpy_dtype:
            self.A = tf.constant(a.T, dtype=self.TF_REAL, name='A')
        elif a.dtype == self.TF_COMPLEX.as_numpy_dtype:
            self.A = tf.constant(a.T, dtype=self.TF_COMPLEX, name='A')
        else:
            raise TypeError('Type of A not supported.'
                            'Only {} and {} supported, A is {}'.format(self.TF_REAL.as_numpy_dtype,
                                                                       self.TF_COMPLEX.as_numpy_dtype,
                                                                       a.dtype))
        if x.dtype != a.dtype:
            raise TypeError('x does not have the same type as A'
                            'Type of x: {}. Type of A: {}'.format(x.dtype, a.dtype))
        if y.dtype != a.dtype:
            raise TypeError('y does not have the same type as A'
                            'Type of y: {}. Type of A: {}'.format(y.dtype, a.dtype))

        # Save dataset
        self.X = x
        self.Y = y

        if shrink_func in ('bg', 'bg2', 'bg3'):
            self.p_a = np.count_nonzero(x) / x.size

        # Declare placeholders for training data
        self.x = tf.placeholder(self.A.dtype, [None, n], name='X')
        self.y = tf.placeholder(self.A.dtype, [None, m], name='Y')

        if shrink_func not in self.shrink_funcs.keys():
            raise ValueError('Unrecognised shrinkage functions. Implemented shrinkage functions are: '
                             '{}'.format([key for key in self.shrink_funcs.keys()]))
        for i, th in enumerate(theta):
            if th.size != self.shrink_funcs[shrink_func]['nparams']:
                raise ValueError('The number of parameters does not match for at least one layer.'
                                 '{} requires {} parameters input has {} parameters at position {}'
                                 '.'.format(shrink_func, self.shrink_funcs[shrink_func]['nparams'], th.size, i))

        self.shrink = self.shrink_funcs[shrink_func]['function']

        # Initialize learnable parameters
        with tf.name_scope('Theta'):
            self._B = self._b_set(b)
            self._theta = self._theta_set(theta)
            self.train_list = []
            # Update trainable list accordingly
            if type(b_train) is bool:
                if b_train:
                    for b_t in self._B:
                        self.train_param(b_t)
                else:
                    for b_t in self._B:
                        self.freeze_param(b_t)
            else:
                for b_t, b_t_train in zip(self._B, b_train):
                    if b_t_train:
                        self.train_param(b_t)
                    else:
                        self.freeze_param(b_t)
            if type(theta_train) is bool:
                if theta_train:
                    for theta_t in self._theta:
                        self.train_param(theta_t)
                else:
                    for theta_t in self._theta:
                        self.freeze_param(theta_t)
            else:
                for theta_t, theta_t_train in zip(self._theta, theta_train):
                    if theta_t_train:
                        self.train_param(theta_t)
                    else:
                        self.freeze_param(theta_t)

        self.iteration = 0

        if self.analyze_data:
            with tf.name_scope('data_stats'):
                x_flat = tf.reshape(self.x, [-1])
                x_nonzero = tf.boolean_mask(x_flat, tf.not_equal(x_flat, tf.zeros_like(x_flat)))
                n_real = tf.constant(n, dtype=self.TF_REAL)
                zeros = n_real - tf.divide(tf.cast(tf.size(x_nonzero), dtype=self.TF_REAL),
                                           tf.constant(batch_size, dtype=self.TF_REAL))
                sparsity = tf.divide(zeros, n_real, name='sparsity')
                tf.summary.scalar('x_sparsity', sparsity)

                signal = tf.matmul(self.x, self.A)
                w = self.y - signal
                e_s = self.squared_norm(signal, axis=-1)
                noise = self.squared_norm(w, axis=-1)
                snr = self.pow2db(tf.reduce_mean(e_s) / tf.reduce_mean(noise), name='snr')
                tf.summary.scalar('SNR', snr)

                self.add_histogram('X', x_nonzero)
                self.add_histogram('Noise', w)
                self.add_histogram('Y', self.y)
                self.add_histogram('A', self.A)
                self.add_histogram('A_cols', tf.reduce_mean(self.A, axis=1))
                self.add_histogram('A_rows', tf.reduce_mean(self.A, axis=0))

        x_seq = self.network(self.y)

        self.NMSE = []

        if nmse:
            with tf.name_scope('NMSE'):
                self.NMSE = [self.nmse(x_t, self.x, name='layer{}'.format(t)) for t, x_t in enumerate(x_seq)]
        else:
            self.NMSE = None

        x_ = x_seq[-1]

        self.L = self.loss(x_, self.x, squared=squared)
        self.train_step = self.optimizer(self.L) if train else None
        self._init = tf.global_variables_initializer()

        if save_support:
            self.support_image(x_, self.x)

        tf.set_random_seed(seed)

    def __enter__(self):
        print("Tensorflow version " + tf.__version__)
        self.session = tf.Session()
        if self.save_results:
            tf.summary.scalar('Loss', self.L)
            if self.NMSE is not None:
                tf.summary.scalar('NMSE', self.NMSE[-1])
            self.summary = tf.summary.merge_all()

        self.session.__enter__()
        if self.save_results:
            self.writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

        self.session.run(self._init)

    def train_param(self, param):
        if param not in tf.trainable_variables():
            raise ValueError('Passed parameter is not a trainable variable')
        if param not in self.train_list:
            self.train_list.append(param)

    def freeze_param(self, param):
        if param not in tf.trainable_variables():
            raise ValueError('Passed parameter is not a trainable variable')
        if param in self.train_list:
            self.train_list.remove(param)

    @classmethod
    def _2complex(cls, x, c_type='r', name='num2complex'):
        """
        Real to complex tensorflow casting that supports backpropagation
        @param x: tensor to be casted
        @param c_type: tensor type
        @param name: Optional name
        @return: Complex-casted tensor
        """

        mat_types = (('r', 'real'), ('c', 'complex'), ('s', 'split'))

        with tf.name_scope(name):
            if c_type in mat_types[0]:
                x_ = tf.complex(x, tf.zeros(1, dtype=cls.TF_REAL, name='0'))
            elif c_type in mat_types[1]:
                x_ = x
            elif c_type in mat_types[2]:
                real, im = tf.split(x, 2, axis=-1)
                x_ = tf.complex(tf.squeeze(real, axis=-1), tf.squeeze(im, axis=-1))
            else:
                raise ValueError('Specified tensor type not recognised. Supported tensor types are '
                                 + str(mat_types))
        return x_

    def support_image(self, x_, x, max_outputs=1):
        """
        square_height = int(np.sqrt(self.N * 2) // 2 + 1)
        shape = [-1, self.N, 2]
        for fact in np.arange(square_height, 0, -1):
            if self.N % fact == 0:
                shape = [-1, self.N // fact, 2 * fact]
                break
        new_shape = tf.constant(shape, dtype=tf.int32)
        """
        with tf.name_scope('Image_reconstruction'):
            aspect_ratio = 5
            new_width = int(self.N / aspect_ratio)
            new_size = tf.constant([self.N, new_width], dtype=tf.int32)

            bright_max = tf.constant(1.0, dtype=self.TF_REAL)
            bright_min = tf.constant(0.95, dtype=self.TF_REAL)

            x_x_ = tf.stack([x, x_], axis=-1)
            # x_x_ = tf.reshape(x_x_, new_shape)

            support = tf.math.not_equal(x_x_, tf.zeros_like(x_x_), name='support')

            hue = tf.floormod(tf.angle(x_x_) / tf.constant(2 * np.pi, dtype=self.TF_REAL),
                              tf.constant(1.0, dtype=self.TF_REAL), name='hue')
            saturation = tf.divide(tf.abs(x_x_), tf.reduce_max(tf.abs(x_x_), keepdims=True),
                                   name='saturation')
            bright = (bright_min - bright_max) * tf.cast(support, dtype=self.TF_REAL, name='value') + bright_max

            hsv = tf.stack([hue, saturation, bright], axis=-1, name='hsv')
            rgb = hsv_to_rgb(hsv, name='rgb')
            rgb = resize_nearest_neighbor(rgb, new_size)

            tf.summary.image('original_vs_reconstruction', rgb, max_outputs=max_outputs)

    @staticmethod
    def _split_complex(x):
        return np.stack((x.real, x.imag), axis=-1)

    @staticmethod
    def _merge_complex(x):
        real, im = np.split(x, 2, axis=-1)
        return real.squeeze(axis=-1) + im.squeeze(axis=-1) * 1j

    def layer(self, x_current, v_current, y, t, name='LAMP_t'):
        """

        @param x_current: complex input reconstructed signals at t-th layer
        @param v_current: complex input residual errors at t-th layer
        @param y: complex measured signals
        @param t: Layer index
        @param name: Tensorflow name
        @return: reconstructed signals and residual errors for the next iteration
        """
        with tf.name_scope(name):

            if y.dtype == self.TF_COMPLEX:
                b_t = self._2complex(self._B[t], c_type='s', name='B_t')
                m_ = tf.constant(self.M, dtype=self.TF_COMPLEX, name='M')
            else:
                b_t = tf.identity(self._B[t], name='B_t')
                m_ = tf.constant(self.M, dtype=self.TF_REAL, name='M')

            r = tf.add(x_current, tf.matmul(v_current, b_t), name='r_t')
            sigma_t = tf.divide(tf.cast(tf.norm(v_current, axis=-1, keepdims=True), dtype=self.TF_REAL, name='v_norm'),
                                tf.sqrt(tf.constant(self.M, dtype=self.TF_REAL)), name='sigma_t')

            x_next, b_onsager = self.shrink(r, sigma_t, self._theta[t], m_, name="x_tplus1")

            onsager = tf.multiply(b_onsager, v_current, name='onsager')
            v_next = tf.add(tf.subtract(y, tf.matmul(x_next, self.A), name='res_error'), onsager,
                            name='v_tplus1')

        return x_next, v_next

    def network(self, y):
        """

        Unfolding of AMP by concatenating num_layers LAMP layers

        @param y: measured signals
        @return: final reconstructed signals
        """

        with tf.name_scope('LAMP' + str(self.T)):
            v_t0 = tf.identity(y, name='v0')
            x_t0 = tf.zeros((1, self.N), dtype=self.x.dtype, name='x0')
            x_seq = []
            for t in range(self.T):
                name = 'layer_' + str(t)
                x_t1, v_t1 = self.layer(x_t0, v_t0, y, t, name=name)
                x_seq.append(x_t1)
                x_t0 = x_t1
                v_t0 = v_t1
        return x_seq

    @staticmethod
    def pow2db(x, name='dB'):
        with tf.name_scope(name):
            db = 10 * tf.log(x) / np.log(10)
        return db

    @classmethod
    def squared_norm(cls, x, axis=-1, real=True, keepdims=None, name='squared_norm'):
        with tf.name_scope(name):
            x_squared = tf.multiply(x, tf.conj(x))
            sq_norm = tf.reduce_sum(x_squared, axis=axis, keepdims=keepdims)
            if real:
                sq_norm = tf.cast(sq_norm, dtype=cls.TF_REAL)
        return sq_norm

    @classmethod
    def loss(cls, x_, x, name='Loss', squared=True):
        with tf.name_scope(name):
            x_err = tf.subtract(x_, x, name='x_err')
            norm_err = cls.squared_norm(x_err, name='squared_err') if squared else tf.abs(x_err, name='L1')
            loss_t = tf.reduce_mean(norm_err, name='L_T')
        return loss_t

    @classmethod
    def nmse(cls, x_, x, db=True, name='NMSE', tol=1e-8):
        with tf.name_scope(name):
            x_err = tf.subtract(x_, x, name='x_err')
            sq_err = cls.squared_norm(x_err, name='squared_err')
            sq_norm = cls.squared_norm(x, name='squared_norm')
            norm_sq_err = tf.divide(sq_err, sq_norm, name='norm_squared_err')
            nmse = tf.reduce_mean(norm_sq_err, name='nmse')
            if tol is not None:
                nmse += tol
            if db:
                nmse = cls.pow2db(nmse)
        return nmse

    def optimizer(self, cost, name='Optimization'):
        with tf.name_scope(name):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=self.train_list)
        return train_step

    def _theta_set(self, theta):
        _theta = []
        for t, theta_t_init in enumerate(theta):
            theta_name = 'theta' + str(t)
            theta_t = tf.Variable(theta_t_init, dtype=self.TF_REAL, name=theta_name)
            for i in range(theta_t_init.size):
                tf.summary.scalar('_'.join((theta_name, str(i+1))), theta_t[i])
            _theta.append(theta_t)
        return _theta

    def _b_set(self, b):
        _B = []
        if self.tied:
            b_t = b[0]
            b_type = b_t.dtype
            b_init = b_t.T
            if b_type == self.TF_REAL.as_numpy_dtype:
                b_0 = tf.Variable(b_init, dtype=self.TF_REAL, name='B_0')
                self.add_histogram('B_0', b_0)
            elif b_type == self.TF_COMPLEX.as_numpy_dtype:
                b_init = self._split_complex(b_init)
                b_0 = tf.Variable(b_init, dtype=self.TF_REAL, name='B_0')
                b_complex = self._2complex(b_0, c_type='s', name='B_complex')
                self.add_histogram('B_0', b_complex)
            else:
                raise TypeError('Unsupported type {} for B'.format(b_type))
            _B = [b_0] * self.T
        else:
            for t, b_t in enumerate(b):
                b_name = 'B' + str(t)
                b_type = b_t.dtype
                b_init = b_t.T
                if b_type == self.TF_REAL.as_numpy_dtype:
                    b_ = tf.Variable(b_init, dtype=self.TF_REAL, name=b_name)
                    self.add_histogram(b_name, b_)
                elif b_type == self.TF_COMPLEX.as_numpy_dtype:
                    b_init = self._split_complex(b_init)
                    b_ = tf.Variable(b_init, dtype=self.TF_REAL, name=b_name)
                    b_complex = self._2complex(b_, c_type='s', name=b_name+'_complex')
                    self.add_histogram(b_name, b_complex)
                else:
                    raise TypeError('Unsupported type {} for {}'.format(b_type, b_name))
                _B.append(b_)
        return _B

    @property
    def theta(self):
        if self.session is not None:
            return self.session.run(self._theta)
        else:
            raise RuntimeError('learnable parameters are only available within an open session')

    @theta.setter
    def theta(self, theta):
        if self.session is not None:
            for th_tensor, th_val in zip(self._theta, theta):
                th_tensor.load(th_val, self.session)
        else:
            raise RuntimeError('learnable parameters are only available within an open session')

    @property
    def b(self):
        if self.session is not None:
            b_ = self._B
            b_out = self.session.run(b_)
            b = []
            for b_t in b_out:
                b_t_ = self._merge_complex(b_t) if self.A.dtype == self.TF_COMPLEX else b_t
                b.append(b_t_.T)
            return b
        else:
            raise RuntimeError('learnable parameters are only available within an open session')

    @b.setter
    def b(self, b):

        if self.session is not None:
            b_ = [b[0]] * self.T if self.tied else b
            for b_th, b_val in zip(self._B, b_):
                b_type = b_val.dtype
                b_val = b_val.T
                if b_type == self.TF_COMPLEX.as_numpy_dtype:
                    b_val = self._split_complex(b_val)
                elif b_type != self.TF_REAL.as_numpy_dtype:
                    raise TypeError('Unsupported type {}'.format(b_type))
                b_th.load(b_val, self.session)
        else:
            raise RuntimeError('learnable parameters are only available within an open session')

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Closing Tensorflow session')
        if self.writer is not None:
            self.writer.close()
        self.session.close()
        self.session.__exit__(exc_type, exc_val, exc_tb)
        self.session = None

    def training_step(self, loss=True, nmse=False, update_log=False):

        x_batch, y_batch = self.batcher.next_batch(self.X, self.Y)
        inputs = {self.x: x_batch, self.y: y_batch}
        outputs = [self.train_step]
        ret = []
        if loss:
            outputs.append(self.L)
        if nmse:
            if self.NMSE is None:
                raise RuntimeError('NMSE not included in graph')
            outputs.append(self.NMSE[-1])
        if update_log:
            outputs.append(self.summary)
        out = self.session.run(outputs, feed_dict=inputs)
        self.iteration += 1
        if update_log:
            summary = out[outputs.index(self.summary)]
            self.writer.add_summary(summary, self.iteration)
        if loss:
            ret.append(out[outputs.index(self.L)])
        if nmse:
            ret.append(out[outputs.index(self.NMSE[-1])])
        if ret:
            return ret

    def get_nmse(self):
        x_batch, y_batch = self.batcher.next_batch(self.X, self.Y)
        inputs = {self.x: x_batch, self.y: y_batch}
        out = self.session.run(self.NMSE[-1], feed_dict=inputs)
        return out

    def nmse_seq(self):
        x_batch, y_batch = self.batcher.next_batch(self.X, self.Y)
        inputs = {self.x: x_batch, self.y: y_batch}
        return np.array(self.session.run(self.NMSE, feed_dict=inputs))

    @classmethod
    def add_histogram(cls, name, x):
        if x.dtype == cls.TF_REAL:
            tf.summary.histogram(name, x)
        elif x.dtype == cls.TF_COMPLEX:
            with tf.name_scope(name):
                tf.summary.histogram('amplitude', tf.abs(x))
                tf.summary.histogram('phase', tf.angle(x))
        else:
            raise TypeError('Variable has the unsupported type {}'.format(x.dtype))

    @classmethod
    def scaled_soft_th(cls, r, sigma, theta, m, name='eta', constrained=False):
        """
        @param r: Complex input matrix
        @param sigma: Noise standard deviation estimate
        @param theta: Learnable parameters beta and alpha
        @param name: Tensorflow name for the output
        @param m: Dimension of y for Onsager correction
        @param constrained: Whether the values of the function should be constrained
        @return: Complex elementwise soft-thresholded r and onsager factor
        """

        with tf.name_scope('soft_th'):

            if constrained:
                beta = cls.constrained_theta(theta[0], name='beta')
            else:
                beta = tf.identity(theta[0], name='beta')

            if r.dtype == cls.TF_COMPLEX:
                beta = cls._2complex(beta)

            alpha = tf.identity(theta[1], name='alpha')
            threshold = tf.multiply(alpha, sigma, name='lambda')

            amplitude = tf.abs(r, name='abs')
            sign = tf.sign(r, name='sign')
            shrink = tf.multiply(sign, tf.cast(tf.nn.relu(amplitude - threshold), dtype=sign.dtype), name='shrink')
            eta = tf.multiply(beta, shrink, name=name)
            x_0_norm = tf.count_nonzero(eta, axis=-1, keepdims=True, dtype=eta.dtype, name='x_0_norm')
            b = tf.multiply(beta / m, x_0_norm, name='b_tplus1')

        return eta, b

    @staticmethod
    def constrained_theta(theta, name=None):
        return tf.sigmoid(theta, name=name)

    @classmethod
    def piecewise_linear(cls, r, sigma, theta, m, name='eta', constrained=False):
        """
        @param r: Complex input matrix
        @param sigma: Noise standard deviation estimate
        @param theta: Learnable parameters beta and alpha
        @param name: Tensorflow name for the output
        @param m: Dimension of y for Onsager correction
        @param constrained: Whether the values of the function should be constrained
        @return: Complex elementwise piecewise-thresholded r and onsager factor
        """

        def cast(x):
            return tf.cast(x, dtype=r.dtype)

        with tf.name_scope('pw_lin'):

            theta1 = theta[0]
            theta2 = theta[1]
            theta3 = theta[2]
            theta4 = theta[3]
            theta5 = theta[4]

            if constrained:
                theta3 = cls.constrained_theta(theta3, name='slope3')
                theta4 = cls.constrained_theta(theta4, name='slope4')
                theta5 = cls.constrained_theta(theta5, name='slope5')

            threshold1 = tf.multiply(theta1, sigma, name='threshold_1')
            threshold2 = tf.multiply(theta2, sigma, name='threshold_2')

            amplitude = tf.abs(r, name='abs')
            sign = tf.sign(r, name='sign')

            mask_small = tf.less(amplitude, threshold1, name='bmask1')
            mask_large = tf.less(threshold2, amplitude, name='bmask2')

            eta_small = tf.multiply(cast(theta3), r, name='eta_small')
            eta_medium = theta4 * (amplitude - threshold1) + theta3 * threshold1
            eta_medium = tf.multiply(sign, cast(eta_medium), name='eta_medium')
            eta_large = theta5 * (amplitude - threshold2) + theta4 * (threshold2 - threshold1) + theta3 * threshold1
            eta_large = tf.multiply(sign, cast(eta_large), name='eta_large')

            shrink = tf.where(mask_small, eta_small, eta_medium)
            eta = tf.where(mask_large, eta_large, shrink, name=name)

            count_small = tf.count_nonzero(mask_small, axis=-1, keepdims=True, dtype=eta.dtype)
            count_large = tf.count_nonzero(mask_large, axis=-1, keepdims=True, dtype=eta.dtype)
            count_medium = cast(tf.shape(r)[-1]) - count_small - count_large

            b = tf.divide(cast(theta3) * count_small + cast(theta4) * count_medium
                          + cast(theta5) * count_large, m, name='b_tplus1')

            # b = tf.divide(tf.reduce_sum(tf.gradients(eta, r, colocate_gradients_with_ops=True),
            #                          axis=-1, keepdims=True), m, name='b_tplus1')

        return eta, b

    @classmethod
    def exponential(cls, r, sigma, theta, m, name='eta', constrained=False):
        """
        @param r: Complex input matrix
        @param sigma: Noise standard deviation estimate
        @param theta: Learnable parameters beta and alpha
        @param name: Tensorflow name for the output
        @param m: Dimension of y for Onsager correction
        @param constrained: Whether the values of the function should be constrained
        @return: Complex elementwise exponential-thresholded r and onsager factor
        """
        def cast(x, cast_name):
            return tf.cast(x, dtype=r.dtype, name=cast_name)
        
        with tf.name_scope('exp'):
            
            theta1 = cast(theta[0], cast_name='theta1')
            theta2 = cast(theta[1], cast_name='theta2')
            theta3 = cast(theta[2], cast_name='theta3')
            if constrained:
                theta2 = cls.constrained_theta(theta2, name='slope2')

            sigma_ = cast(sigma, cast_name='sigma')
            two = tf.constant(2.0, dtype=r.dtype, name='2')
            c = tf.pow(sigma_ * theta1, two, name='c')

            lin_term = tf.multiply(theta2, r, name='lin_term')
            r2 = tf.multiply(r, tf.conj(r), name='r_sq')
            exp = tf.exp(-tf.divide(r2, two * c), name='exp')
            exp_term = tf.multiply(theta3 * r, exp, name='exp_term')

            eta = tf.add(lin_term, exp_term, name=name)
            diff = tf.add(theta2, tf.divide(theta3 * exp * (c - r2), c), name='diff_eta')
            b = tf.divide(tf.reduce_sum(diff, axis=-1, keepdims=True), m, name='b_tplus1')

        return eta, b

    @staticmethod
    def _bg(r, sigma, p_ratio, phi, m, name='eta'):
        """
        @param r: Complex input matrix
        @param sigma: Noise standard deviation estimate
        @param phi: Signal std deviation
        @param p_ratio: ratio between prob of s=0 and s=1
        @param name: Tensorflow name for the output
        @param m: Dimension of y for Onsager correction
        @return: Complex elementwise exponential-thresholded r and onsager factor
        """

        one = tf.constant(1.0, dtype=r.dtype, name='1')
        two = tf.constant(2.0, dtype=r.dtype, name='2')
        s2 = tf.pow(sigma, two, name='sigma2')
        r2 = tf.multiply(r, tf.conj(r), name='r2')

        a = tf.add(one, tf.divide(s2, phi), name='1pluss2_phi')
        b = tf.multiply(p_ratio, tf.sqrt(one + phi / s2), name='sqrt1plusphi_s2')
        exp = tf.exp(-tf.divide(r2, two * s2 * a), name='exp')

        dem = tf.multiply(a, 1 + b * exp, name='denom')

        eta = tf.divide(r, dem, name=name)
        with tf.name_scope('diff'):
            diff = tf.divide(one + tf.divide(b * r2 * exp, dem * s2), dem, name='diff_eta')
        b = tf.divide(tf.reduce_sum(diff, axis=-1, keepdims=True), m, name='b_tplus1')

        return eta, b

    @classmethod
    def bg(cls, r, sigma, theta, m, x_p=None, name='eta', constrained=False):
        """
        @param r: Complex input matrix
        @param sigma: Noise standard deviation estimate
        @param theta: Learnable parameters beta and alpha
        @param name: Tensorflow name for the output
        @param m: Dimension of y for Onsager correction
        @param x_p: Prior information
        @param constrained: Whether the values of the function should be constrained
        @return: Complex elementwise exponential-thresholded r and onsager factor
        """

        gamma_tol = .01

        def cast(x, cast_name):
            return tf.cast(x, dtype=r.dtype, name=cast_name)

        with tf.name_scope('bg'):
            phi = cast(theta[0], cast_name='theta1')
            s = cast(sigma, cast_name='sigma')
            theta_ratio = cast(theta[1], cast_name='theta2')
            if x_p is None:
                p_ratio = theta_ratio
            else:
                if tf.contrib.framework.is_tensor(x_p):
                    with tf.name_scope('support_estimate'):
                        gamma = tf.tanh(tf.abs(x_p), name='gamma_unbound')
                        gamma_max = tf.constant(1 - gamma_tol, dtype=gamma.dtype, name='gamma_max')
                        gamma_min = tf.constant(gamma_tol, dtype=gamma.dtype, name='gamma_min')
                        gamma = tf.clip_by_value(gamma, gamma_min, gamma_max, name='gamma_clipped')
                        gamma = cast(gamma, cast_name='gamma')
                else:
                    gamma = tf.constant(x_p, dtype=r.dtype, name='gamma')
                ones = tf.ones_like(gamma)
                gamma_ratio = tf.divide(ones - gamma, gamma, name='gamma_ratio')
                p_ratio = tf.multiply(theta_ratio, gamma_ratio)
            if constrained:
                pass
            return cls._bg(r, s, p_ratio, phi, m, name=name)

    shrink_funcs = {'scaled_soft_th': {'function': scaled_soft_th,      'nparams': 2},
                    'pw_lin':         {'function': piecewise_linear,    'nparams': 5},
                    'exp':            {'function': exponential,         'nparams': 3}}

    @classmethod
    def shrink(cls, x, theta, shrink_func, sigma=1):
        if shrink_func not in cls.shrink_funcs.keys():
            raise ValueError('Unrecognised shrinkage functions. Implemented shrinkage functions are: '
                             '{}'.format([key for key in cls.shrink_funcs.keys()]))

        if theta.size != cls.shrink_funcs[shrink_func]['nparams']:
            raise ValueError('The number of parameters does not match.'
                             '{} requires {} parameters input has {} parameters.'
                             '.'.format(shrink_func, cls.shrink_funcs[shrink_func]['nparams'], theta.size))

        in_tensor = tf.constant(x)
        out_tensor, _ = cls.shrink_funcs[shrink_func]['function'](in_tensor, sigma, theta, m=1)

        with tf.Session() as sess:
            y = sess.run(out_tensor)
        return y

    class Batcher:
        def __init__(self, batch_size, train_size, shuffle=True):
            self.batch_size = batch_size
            self.train_size = train_size
            self.shuffle = shuffle
            self.indices = np.arange(train_size)
            self.iter = 0
            if shuffle:
                np.random.shuffle(self.indices)

        def next_batch(self, *args, batch_size=None, shuffle=None):
            if batch_size is None:
                batch_size = self.batch_size
            if shuffle is None:
                shuffle = self.shuffle
            next_iter = self.iter + batch_size
            if next_iter >= self.train_size:
                if next_iter > self.train_size and shuffle:
                    raise Warning('Batch size overruns epoch limit. Shuffling may cause repeated data within the batch')
                batch = [x[self.indices[self.iter:],] for x in args]
                self.iter = next_iter % self.train_size
                if shuffle:
                    np.random.shuffle(self.indices)
                tail_batch = [x[self.indices[:self.iter],] for x in args]
                batch = [np.concatenate((a, b), axis=0) for a, b in zip(batch, tail_batch)]
            else:
                batch = [x[self.indices[self.iter:next_iter],] for x in args]
                self.iter = next_iter
            return tuple(batch)


def _penalty_soft_th(theta):

    with tf.name_scope('penalty_soft_th'):
        beta = theta[0]
        penalty = tf.nn.relu(tf.abs(beta - 0.5) - 0.5)

    return penalty


def _penalty_pw_lin(theta):

    with tf.name_scope('penalty_pw_lin'):

        slopes = theta[2:]
        penalties = tf.nn.relu(tf.abs(slopes - 0.5) - 0.5)
        penalty = tf.reduce_sum(penalties)

    return penalty


def _penalty_exp(theta):
    with tf.name_scope('penalty_exp'):
        slopes = tf.stack([theta[1], theta[1] + theta[2]])
        penalties = tf.nn.relu(tf.abs(slopes - 0.5) - 0.5)
        penalty = tf.reduce_sum(penalties)

    return penalty


LAMP.shrink_funcs = {'scaled_soft_th': {'function': LAMP.scaled_soft_th, 'nparams': 2,
                                        'default_theta': [np.ones(2)], 'penalty': _penalty_soft_th},
                     'pw_lin':         {'function': LAMP.piecewise_linear, 'nparams': 5,
                                        'default_theta': [np.array([0.75, 1.25, 0.05, 0.5, 1])],
                                        'penalty': _penalty_pw_lin},
                     'exp':            {'function': LAMP.exponential, 'nparams': 3,
                                        'default_theta': [np.array([1, 1, -1])], 'penalty': _penalty_exp},
                     'bg':              {'function': LAMP.bg, 'nparams': 2,
                                         'default_theta': [np.ones(2)]}}


class EarlyStopper:
    def __init__(self, patience, delta, max_iterations, warmup=0):
        self.patience = patience
        self.delta = delta
        self.max_iter = max_iterations
        self.warmup = warmup
        self.best_val = None
        self.best_model = None
        self.best_iter = 0
        self.iter = 0
        self.cnt = 0

    def reset(self, best=None, patience=None, delta=None, max_iterations=None, model=None):
        if patience is not None:
            self.patience = patience
        if delta is not None:
            self.delta = delta
        if max_iterations is not None:
            self.max_iter = max_iterations
        self.best_val = best
        self.best_model = model
        self.best_iter = 0
        self.iter = 0
        self.cnt = 0

    def _update_best(self, val, i, model):
        self.best_val = val
        self.best_iter = i
        self.best_model = model

    def should_continue(self, val, model=None):
        self.iter += 1
        if self.iter > self.max_iter:
            return False
        elif self.best_val is None or val is None:
            self._update_best(val, self.iter - 1, model)
            self.cnt = 0
            return True
        elif self.iter < self.warmup:
            self._update_best(val, self.iter - 1, model)
            self.cnt = 0
            return True
        elif (self.best_val - val) > self.delta:
            self._update_best(val, self.iter - 1, model)
            self.cnt = 0
            return True
        else:
            self.cnt += 1
            if self.cnt > self.patience:
                return False
            else:
                return True

    def best(self):
        return {'val': self.best_val, 'iter': self.best_iter, 'model': self.best_model}
