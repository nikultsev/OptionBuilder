# conda install -n environment package
# conda info --env
# conda create -name new_environment
# conda list -n environment
import numpy as np
import matplotlib.pyplot as plt
import sympy
from zoom_function import zoom_factory, rebind_pan_to_middle_click
from derivative_classes import *  # ignores __main__ still

# Plan: test the payout of different (individual or combined) options for a fixed time on the S and P 500 (or other
# radomly generated ones). I want the ability to set the point in time we are beginning
# the strategy from, imput the expiritation(s) of the strategy, and optinally block out
# future S and P 500 trading data


def generate_paths(S0, r, sigma, T, M, I):
    # S0 initial stock value, r risk free rate, sigma volatility, T final time horizon, M number of time steps, I number of paths to be generated
    dt = T/M
    paths = np.zeros((M+1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean())/rand.std()
        paths[t] = paths[t - 1] * \
            np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)

    # must return ndarray - shape (M+1, I) # ndarray is n dimensional array
    return paths


def main():

    dtype = [('Date', 'U10'), ('Close/Last', 'f8'),
             ('Open', 'f8'), ('High', 'f8'), ('Low', 'f8')]
    data = np.genfromtxt('HistoricalData_1717244350919.csv',
                         delimiter=',', dtype=dtype, names=True, encoding='utf-8')

    fig, axs = plt.subplots(2, 3)

    S0 = 100.
    r = 1
    sigma = 0.5
    T = 1.
    M = 2531
    I = 10

    # print(np.shape(paths))

    open_prices = data['Open']
    close_prices = data['CloseLast']

    interlaced = np.empty(
        (open_prices.size + close_prices.size), dtype=open_prices.dtype)
    interlaced[0::2] = open_prices
    interlaced[1::2] = close_prices

    double_date = np.repeat(data['Date'], 2)
    interlaced[interlaced == 0] = np.nan
    print(np.shape(np.flip(interlaced)))
    S0 = np.flip(interlaced)[0]

    paths = generate_paths(S0, r, sigma, T, M, I)
    axs[0, 1].plot(paths[:, :10], linestyle='dashed')
    axs[0, 1].plot(np.flip(double_date), np.flip(
        interlaced), label='S&P 500', color='black')
    print(np.shape(paths[0]))
    axs[0, 1].set_xticks([])

    zoom_factory(axs[0, 1])
    rebind_pan_to_middle_click()

    plt.show()
    fig.savefig('savedfig.png')


if __name__ == "__main__":
    main()
