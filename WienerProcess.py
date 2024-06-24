# conda install -n environment package
# conda info --env
# conda create -name new_environment
# conda list -n environment
import numpy as np
import matplotlib.pyplot as plt
import sympy
from zoom_function import zoom_factory, rebind_pan_to_middle_click
from derivative_classes import *  # ignores __main__ still
from tqdm import tqdm

# Plan: test the payout of different (individual or combined) options for a fixed time on the S and P 500 (or other
# radomly generated ones). I want the ability to set the point in time we are beginning
# the strategy from, imput the expiritation(s) of the strategy, and optinally block out
# future S and P 500 trading data

dtype = [('Date', 'U10'), ('Close/Last', 'f8'),
         ('Open', 'f8'), ('High', 'f8'), ('Low', 'f8')]
data = np.genfromtxt('HistoricalData_1717244350919.csv',
                     delimiter=',', dtype=dtype, names=True, encoding='utf-8')
yearly_trading_days = 252
open_prices = data['Open']
close_prices = data['CloseLast']
average_daily_prices_10_years = (open_prices + close_prices)/2


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


def evolve_stock_market(dataset, start_date, expiry_date, option_product):
    # choose start date, stock dataset,option_product, and its expiry and observe the
    # value of your position evolve as we progress along the dataset

    pass


def main():

    def call_price_ledger(day):

        # calculate price of calls for the 252nd trade day, 5 percent increments

        day = day-1

        test_volatility = np.std(average_daily_prices_10_years[0:251])

        spot_price = average_daily_prices_10_years[day]

        option_class = OptionPrices(S=spot_price, r=0.03,
                                t=1, T=1.25, sigma=test_volatility)
        
        strikes = np.linspace(0 * spot_price, 10 * spot_price, 20)

        prices = np.empty((len(strikes), 1))

        for i, strike in enumerate(strikes):
            print(strike,option_class.CallOption(E = strike)) # price is same
            prices[i] = option_class.CallOption(E = strike)
            
        print(prices)



        

        print(np.shape(average_daily_prices_10_years))

        pass

    call_price_ledger(252)

    def call_and_put_play():

        pass

    def open_close_average():

        fig, axs = plt.subplots(2, 3)

        axs[0, 1].plot(np.linspace(1, 252, 251), np.flip(
            average_daily_prices_10_years[0:yearly_trading_days-1]))

        zoom_factory(axs[0, 1])
        rebind_pan_to_middle_click()
        plt.show()

        #

        # print(np.shape(open_prices), np.shape(close_prices), np.shape(average_daily_price))

    def open_close_diff():
        dtype = [('Date', 'U10'), ('Close/Last', 'f8'),
                 ('Open', 'f8'), ('High', 'f8'), ('Low', 'f8')]
        data = np.genfromtxt('HistoricalData_1717244350919.csv',
                             delimiter=',', dtype=dtype, names=True, encoding='utf-8')

        fig, axs = plt.subplots(2, 3)

        # S0 = 100.
        # r = 1
        # sigma = 0.5
        # T = 1.
        # M = 2531
        # I = 10

        # print(np.shape(paths))

        yearly_trading_days = 252

        open_prices = data['Open']
        close_prices = data['CloseLast']

        interlaced = np.empty(
            (open_prices.size + close_prices.size), dtype=open_prices.dtype)
        interlaced[0::2] = open_prices
        interlaced[1::2] = close_prices

        double_date = np.repeat(data['Date'], 2)
        interlaced[interlaced == 0] = np.nan
        print(np.shape(np.flip(interlaced)))
        # S0 = np.flip(interlaced)[0]

        # paths = generate_paths(S0, r, sigma, T, M, I)
        # axs[0, 1].plot(paths[:, :10], linestyle='dashed')
        axs[0, 1].plot(np.flip(double_date), np.flip(
            interlaced), label='S&P 500', color='black')
        # print(np.shape(paths[0]))
        axs[0, 1].set_xticks([])

        zoom_factory(axs[0, 1])
        rebind_pan_to_middle_click()

        plt.show()
        fig.savefig('savedfig.png')


if __name__ == "__main__":
    main()
