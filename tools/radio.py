import numpy as np

from numpy.random import uniform, binomial, normal, randint  # , exponential


def pow2db(z):
    return 10 * np.log10(np.abs(z))


def db2pow(db):
    return 10 ** (db / 10)


def model(n_users, m_symbols, p_ua, energy, sigma_v, a=None, real=False):

    """

    Generate matrices to build the received signal at a RRH r

    @param n_users: Number of total User Equipments
    @param m_symbols: Number of Symbols in the identification signature
    @param p_ua: User Activity probability 0<=p_ua<=1
    @param energy: Energy constraint for the identification signature
    @param sigma_v: Standard deviation of the noise
    @param a: Matrix A, if already calculated
    @param real: Whether generated data is real or complex
    @return: Signature matrix A, Bernoulli-Gaussian vector x_r and noise vector v_r

    """

    #  fade_exp_lambda = 1
    # s_raw = normal(0, 1 / 2, (m_symbols, n_users)) + normal(0, 1 / 2, (m_symbols, n_users)) * 1j
    # energies = np.sum(s_raw * s_raw.conjugate(), axis=0)
    # e_mean = energies.mean()
    # s = s_raw * np.sqrt(energy / e_mean)

    gaussian = awgn if real else cn

    if a is None:

        sigma_s = np.sqrt(energy / m_symbols)
        s = gaussian(sigma_s, (m_symbols, n_users))

        gamma = np.ones(n_users)  # np.exp(- exponential(fade_exp_lambda, n))

        a = s @ np.diag(gamma)

    h = gaussian(1, n_users)
    v = gaussian(sigma_v, m_symbols)

    ua = binomial(1, p_ua, size=n_users)

    x = ua * h

    return a, x, v


def awgn(sigma, size):
    return normal(0, sigma, size)


def cn(sigma, size):
    sigma_ = sigma / np.sqrt(2)
    return awgn(sigma_, size) + awgn(sigma_, size) * 1j


def qpsk(size):
    return 1j ** randint(4, size=size)


def rice_params(shape, scale):
    nu = np.sqrt(scale * shape / (1 + shape))
    sigma = np.sqrt(scale / (2 * (1 + shape)))
    return nu, sigma


def ar_params(alpha, mu, sigma):
    mu_ar = mu * (1 - alpha)
    sigma_ar = sigma * np.sqrt(1 - alpha ** 2)
    return mu_ar, sigma_ar


def rice_fading(shape, scale, size, real=False, alpha=None):
    nu, sigma = rice_params(shape, scale)
    los_phase = np.exp(1j * uniform(low=0, high=2 * np.pi, size=size))
    rice = nu * los_phase + cn(sigma * np.sqrt(2), size)
    if real:
        rice = np.abs(rice)
    if alpha is not None:
        mu_ar, sigma_ar = ar_params(alpha, nu * los_phase, sigma)
        return rice, mu_ar, sigma_ar
    else:
        return rice


def path_loss(size, radius=100, k=10, loss_exp=2, real=False):

    h = rice_fading(shape=k, scale=1, size=size)
    d = uniform(high=radius, size=size)
    theta = uniform(high=2 * np.pi, size=size)

    g = h / (1 + d ** (loss_exp / 2))
    if not real:
        g = g * np.exp(1j * theta)

    return g


def borgerding(n_users, m_symbols, p_ua, sigma_v, a=None, real=False):

    if a is not None:
        a = qpsk((m_symbols, n_users))

    ua = binomial(1, p_ua, size=n_users)
    gain = path_loss(n_users, real=real)
    x = ua * gain
    v = awgn(sigma_v, m_symbols) if real else cn(sigma_v, m_symbols)

    return a, x, v


def rice_qpsk(n_users, m_symbols, radius=100, k=10, exp=4):

    pilots = qpsk((m_symbols, n_users))
    h = path_loss(n_users, radius=radius, k=k, loss_exp=exp, real=False)
    a = pilots @ np.diag(h)

    return a


def unif_att(n_users, radius, exp):
    max_loss = 1 / (radius ** (exp / 2))
    h = uniform(low=max_loss, high=1, size=n_users)
    return h


def unif_att_qpsk(n_users, m_symbols, radius=100, exp=4):

    pilots = qpsk((m_symbols, n_users))
    h = unif_att(n_users, radius, exp)
    a = pilots @ np.diag(h)

    return a


def unif_att_gauss(n_users, m_symbols, energy, radius=100, exp=4, real=False):

    gaussian = awgn if real else cn

    sigma_s = np.sqrt(energy / m_symbols)
    pilots = gaussian(sigma_s, (m_symbols, n_users))
    h = unif_att(n_users, radius, exp)
    a = pilots @ np.diag(h)

    return a


def mean_uniform_powerloss(radius=100, exp=4):
    d_ro = radius ** exp
    d_ro2 = radius ** (exp/2)
    return (d_ro + d_ro2 + 1) / (3 * d_ro)


def gen_fading(batch_size, a, n, m, p, sigma, k=10, radius=100, exp=4):

    x = np.zeros((batch_size, n), dtype=a.dtype)
    y = np.zeros((batch_size, m), dtype=a.dtype)

    for j in range(batch_size):

        h = rice_fading(shape=k, scale=1, size=n)
        f = unif_att(n, radius, exp)
        ua = binomial(1, p, size=n)

        x_r = h * ua * f
        v_r = cn(sigma, m)

        x[j,] = x_r
        ax_r = a @ x_r
        y[j,] = ax_r + v_r

    return x, y


def gen_data(batch_size, a, n, m, p, energy, sigma, real=False, verbose=False):

    dtype = np.float64 if real else np.complex128

    mean_signal = 0
    mean_noise = 0

    x = np.zeros((batch_size, n), dtype=dtype)
    y = np.zeros((batch_size, m), dtype=dtype)

    for j in range(batch_size):
        _, x_r, v_r = model(n, m, p, energy, sigma, a, real=real)
        x[j, ] = x_r
        ax_r = a @ x_r
        y[j, ] = ax_r + v_r

        if verbose:
            noise = np.sum(v_r * v_r.conj())
            e_a = np.sum(ax_r * ax_r.conj())
            mean_signal += e_a
            mean_noise += noise

    if verbose:
        mean_signal /= batch_size
        mean_noise /= batch_size

        signal_db = pow2db(mean_signal)
        noise_db = pow2db(mean_noise)
        snr = signal_db - noise_db

        non_zeros = np.count_nonzero(x)
        size = batch_size * n
        sparse = (size - non_zeros) / size * 100

        print('Data generated')
        print('\tSparsity level [%]:\t{}\n'
              '\tMean energy of S:\t{}\n'
              '\tMean Noise energy [dB]:\t{}\n'
              '\tMean SNR [dB]:\t\t{}'.format(sparse, mean_signal, noise_db, snr))

    return x, y


def pathloss_energy(loss_exp, max_radius, min_radius=0):

    if loss_exp == 2:
        def integral(x):
            return 0.5 * (x / (1 + x ** 2) + np.arctan(x))
    else:
        raise ValueError('Path loss exponent {} not supported'.format(loss_exp))

    return (integral(max_radius) - integral(min_radius)) / (max_radius - min_radius)


def rice_energy(shape, scale=1):
    nu, sigma = rice_params(shape, scale)
    return 2 * sigma ** 2 + nu ** 2
