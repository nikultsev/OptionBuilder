# conda install -n environment package
# conda info --env
# conda create -name new_environment
# conda list -n environment
import numpy as np
import matplotlib.pyplot as plt
import sympy


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


S0 = 100
r = 0.05
sigma = 0.2
T = 100
M = 50
I = 250000

paths = generate_paths(S0, r, sigma, T, M, I)

plt.plot(paths[:, 10])


def main():

    dtype = [('Date', 'U10'), ('Close/Last', 'f8'),
             ('Open', 'f8'), ('High', 'f8'), ('Low', 'f8'),]
    data = np.genfromtxt('HistoricalData_1717244350919.csv',
                         delimiter=',', dtype=dtype, names=True, encoding='utf-8')

    # data = np.transpose(data)

    print(data.dtype)
    open_prices = data['Open']
    close_prices = data['CloseLast']

    interlaced = np.empty(
        (open_prices.size + close_prices.size), dtype=open_prices.dtype)
    interlaced[0::2] = open_prices

    interlaced[1::2] = close_prices

    double_date = np.repeat(data['Date'], 2)
    interlaced[interlaced == 0] = np.nan

    print(interlaced, len(interlaced), np.shape(interlaced))
    plt.plot(np.flip(double_date), np.flip(interlaced))
    plt.xticks([0])
    # plt.plot(data['Open'])

    plt.show()


if __name__ == "__main__":
    main()
