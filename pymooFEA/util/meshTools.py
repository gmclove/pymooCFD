import numpy as np


def get_coeff_and_NN(x_min, x_max, x_tot, NN_init=100, coef_init=1.001):
    max_it = 40
    it = 0
    thresh = 1e-6
    err = np.inf

    x_0 = x_min
    x_f = x_max
    NN = NN_init
    coef = coef_init
    while it < max_it and err > thresh:
        coef_prev = coef
        NN = int(np.log(1 + x_tot / x_0 * (coef - 1)) / np.log(coef) + 3)
        coef = np.e**(np.log(x_f / x_0) / (NN - 3))
        err = abs(coef_prev - coef)
        it += 1
        # print(it, err, coef, NN)
    return coef, NN
