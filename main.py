import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tools import neural_network as nn, radio

date = time.strftime('%Y_%m_%d_%H_%M')


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_hist(x, dir_path, name, title, sparse=False, ext='pdf'):
    make_dir(dir_path)
    if sparse:
        size = x.size
        x_nonzero = x[x.nonzero()]
        zeros = size - x_nonzero.size
        sparsity = zeros / size
        x = x_nonzero
        title = 'non-0 elements of {}. ' \
                '{}/{} zeros ({}%)'.format(title, zeros,
                                           size, np.round(sparsity * 100, 1))
    if np.iscomplexobj(x):
        pass
        mag = np.abs(x)
        phase = np.angle(x)
        plt.hist(x=mag.flatten(), bins='auto')
        plt.title(title)
        plt.xlabel('magnitude')
        plt.ylabel('counts')
        path = os.path.join(dir_path, '.'.join((name + '_mag', ext)))
        plt.savefig(path)
        plt.close()
        plt.hist(x=phase.flatten(), bins='auto')
        plt.title(title)
        plt.xlabel('phase')
        plt.ylabel('counts')
        path = os.path.join(dir_path, '.'.join((name + '_phase', ext)))
        plt.savefig(path)
        plt.close()

    else:
        plt.hist(x=x.flatten(), bins='auto')
        plt.title(title)
        plt.xlabel('values')
        plt.ylabel('counts')
        path = os.path.join(dir_path, '.'.join((name, ext)))
        plt.savefig(path)
        plt.close()


# Deep Learning parameters
D = 500                # Batch size for training
num_layers = 10         # Maximum layer size T
num_iterations = 2000   # Max number of training iterations
lr = 0.005              # Learning rate for gradient descent
euclidean = True       # Use L2 norm or L1 norm to optimize

real_data = False

# Early stopper parameters
warmup = 0
patience = 100
db_delta = 0.2


def join(*args, sep='_'):
    return sep.join(args)


base_dir = os.path.join(os.getcwd(), 'data', 'log', join(date, 'lamp'))

# Radio parameters
N = 500         # Number of users N
M = 250         # Number of symbols M
p_ua = 0.1        # Probability of User Activity p
p_10 = 0.01     # Probability of an active user to become inactive in the next time step
SNR = 40        # Signal to noise ratio as defined in AMPs paper
R = 100         # Radius of cell
K = 2
pathloss_exp = 2
E_s = 1  # Signature energy E_s
att = radio.mean_uniform_powerloss(R, pathloss_exp)
Sigma_v = np.sqrt(N * att * p_ua * E_s / radio.db2pow(SNR))

shrink_func = 'scaled_soft_th'
tied = False
a_name = "QPSK"

sigm = Sigma_v / np.sqrt(M)
log_dir = os.path.join(base_dir, join(a_name, shrink_func, 'tied' if tied else 'untied'))


def train_lamp(n, m, p, sigma, a, shrinkage,
               b, theta, batch_size, layer_size, learning_rate,
               theta_train, b_train, tied_lamp, analyze_data,
               min_delta, max_iterations, save_path):

    import sys
    sys.path.append(os.getcwd())

    from tools.neural_network import LAMP, EarlyStopper
    from tools.radio import gen_fading

    db_tol = 1

    def gen_data():
        return gen_fading(batch_size, a, n, m, p,
                          sigma, k=K, radius=R, exp=pathloss_exp)

    x_train, w_train = gen_data()

    lamp = LAMP(x_train, w_train, n, m, a, b, theta, batch_size=batch_size, num_layers=layer_size,
                lr=learning_rate, theta_train=theta_train, b_train=b_train, shrink_func=shrinkage,
                save_results=True, save_support=True, analyze_data=analyze_data,
                nmse=True, tied=tied_lamp, log_dir=save_path, squared=euclidean)

    early_stop = EarlyStopper(patience, min_delta, max_iterations, warmup=warmup)
    model = {'B': b, 'theta': theta}

    print('Starting training. Saving results in {}'.format(save_path))
    with lamp:
        nmse_init = lamp.get_nmse()
        print('Initial NMSE:\t{}\tdB'.format(np.round(nmse_init, 1)))
        nmse = nmse_init

        while early_stop.should_continue(nmse, model):
            update = (early_stop.iter % 5 == 1)

            x_train, w_train = gen_data()
            lamp.X = x_train
            lamp.Y = w_train
            nmse, = lamp.training_step(loss=False, nmse=True, update_log=update)
            model = {'B': lamp.b, 'theta': lamp.theta}

            if update:
                print('Iteration {}.\tNMSE:\t{}\tdB'.format(early_stop.iter, np.round(nmse, 1)))

        print('Stopped at iteration {}.\tNMSE:\t{}\tdB'.format(early_stop.iter, np.round(nmse, 1)))

        best = early_stop.best()
        if best['val'] > nmse_init - db_tol:
            b_nmse = nmse_init
            b_i = 1
            b_model = {'theta': theta, 'B': b}
        else:
            b_nmse = best['val']
            b_i = best['iter']
            b_model = best['model']
        print('Taking over parameters at iteration {}.\tNMSE:\t{}\tdB'.format(b_i, np.round(b_nmse, 1)))

        lamp.theta = b_model['theta']
        lamp.b = b_model['B']

        output = {'NMSE': lamp.nmse_seq(),
                  'iterations': early_stop.iter,
                  'theta': b_model['theta'],
                  'B': b_model['B']}

    return output


def save_shrink(shrink, theta, fname, title='lamp', sigma_span=4, num=1000):

    ext = 'pdf'

    x = np.linspace(-sigma_span, sigma_span, num)
    y = nn.LAMP.shrink(x, theta, shrink_func=shrink)

    fig, ax = plt.subplots()
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.plot(x, y)
    ax.set_title('{} for {}'.format(shrink, title))
    ax.set_xlabel(r'$\frac{r}{\sigma}$')
    ax.set_ylabel('$x$')
    ax.set_aspect('equal')
    plt.savefig(join(fname, ext, sep='.'))
    plt.close(fig)


def save_results(res, path):
    nmse = res['NMSE']
    th = res['theta']
    b_ = res['B']

    np.save(os.path.join(path, 'theta'), th)
    np.save(os.path.join(path, 'B'), b_)
    np.save(os.path.join(path, 'nmse'), nmse)


if __name__ == "__main__":

    make_dir(log_dir)

    a_mat = radio.qpsk((M, N)) * np.sqrt(E_s / M)
    a2 = a_mat * a_mat.conj()
    e_mean = np.mean(np.sum(a2, axis=0))
    print('Mean energy of S for {}:\t{}'.format(a_name, np.abs(e_mean)))
    np.save(os.path.join(log_dir, 'A'), a_mat)
    hist = np.sqrt(np.mean(a2, axis=0))
    save_hist(a_mat, log_dir, 'A_hist', 'A')
    save_hist(np.abs(hist), log_dir, 'A_rms', 'RMS of columns of A')

    B = [a_mat.T.conj()]
    thetas = nn.LAMP.shrink_funcs[shrink_func]['default_theta']

    log_path = os.path.join(log_dir, 'lamp1')

    results = train_lamp(n=N, m=M, p=p_ua, sigma=sigm, batch_size=D, learning_rate=lr,
                         min_delta=db_delta, max_iterations=num_iterations,
                         a=a_mat, b=B, theta=thetas, b_train=True, theta_train=True, shrinkage=shrink_func,
                         tied_lamp=tied, layer_size=1, analyze_data=True,
                         save_path=os.path.join(log_path, 'theta'))

    save_results(results, log_path)

    for t in range(2, num_layers + 1):

        log_path = os.path.join(log_dir, 'lamp' + str(t))
        make_dir(log_path)

        theta_inter = results['theta']
        theta_inter.append(theta_inter[-1])
        B = results['B']
        if not tied:
            B.append(B[-1])

        THETA_TRAIN = np.zeros(t, dtype=np.bool)
        THETA_TRAIN[-1] = True

        inter_results = train_lamp(n=N, m=M, p=p_ua, sigma=sigm, batch_size=D, learning_rate=lr,
                                   min_delta=db_delta, max_iterations=num_iterations,
                                   a=a_mat, b=B, theta=theta_inter, b_train=True, theta_train=THETA_TRAIN,
                                   shrinkage=shrink_func, tied_lamp=tied, layer_size=t, analyze_data=False,
                                   save_path=os.path.join(log_path, 'theta'))

        theta_inter = inter_results['theta']
        B = inter_results['B']

        results = train_lamp(n=N, m=M, p=p_ua, sigma=sigm, batch_size=D, learning_rate=lr,
                             min_delta=db_delta, max_iterations=num_iterations,
                             a=a_mat, b=B, theta=theta_inter, b_train=True, theta_train=True,
                             shrinkage=shrink_func, tied_lamp=tied, layer_size=t, analyze_data=False,
                             save_path=os.path.join(log_path, 'all_params'))

        save_results(results, log_path)
