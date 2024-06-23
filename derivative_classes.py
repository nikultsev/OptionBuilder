import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm

# define the costs and payout structures of each option


class OptionPrices:
    def __init__(self, S, r, t, T, sigma):
        self.S = S  # spot
        self.t = t  # current time
        self.T = T  # expiration
        self.r = r  # risk free rate
        self.sigma = sigma  # percentage annualized volatility within last year

    @staticmethod
    def _N(y):  # cum_dist_norm_normal correct
        z = sp.Symbol('z')
        # SymPy's exp function is used for symbolic integration
        integral_result = sp.integrate(sp.exp(-z**2 / 2), (z, -sp.oo, y))

        # Multiplying by the normalization factor for the normal distribution
        normalized_result = 1 / sp.sqrt(2 * np.pi) * integral_result

        return normalized_result.evalf()

    @staticmethod
    def d1(S, E, r, sigma, T, t):
        return (np.log(S/E) + (r + (sigma**2)/2)*(T - t))/(sigma * np.sqrt(T - t))

    @staticmethod
    def d2(S, E, r, sigma, T, t):

        return OptionPrices.d1(S, E, r, sigma, T, t) - sigma*np.sqrt(T - t)

    def Stock(self):
        return self.S

    def Cash(self, payout):  # E is payout at expiration
        E = payout
        def cost(E, r, T, t): return E * np.exp(-r * (T - t))

        return cost(E, self.r, self.T, self.t)

    def BinaryOption(self, E):
        # E is number which S has to exceed for pay

        d_2 = self.d2(self.S, E, self.r, self.sigma, self.T, self.t)

        def cost(r, T, t): return np.exp(-r*(T-t)) * self._N(d_2)

        return cost(self.r, self.T, self.t)

    def CallOption(self, E):

        d_1 = OptionPrices.d1(self.S, E, self.r, self.sigma, self.T, self.t)
        d_2 = OptionPrices.d2(self.S, E, self.r, self.sigma, self.T, self.t)
        return self.S * OptionPrices._N(d_1) - E * np.exp(-self.r * (self.T - self.t)) * OptionPrices._N(d_2)

    def PutOption(self, E):

        return self.CallOption(E) - self.S + E * np.exp(-self.r * (self.T - self.t))


def main():
    # T and t are in years, sigma is percentile

    def stock_test():
        test = OptionPrices(r=0.05, S=100, t=0.5, T=1, sigma=0.1)
        resolution = 30
        costs = np.empty((resolution))
        print(np.shape(costs))
        values = np.linspace(50, 150, resolution)
        for i in tqdm(range(resolution), desc='looping'):

            costs[i] = cost_of_option_to_get_cash_payout_at_T = test.Stock(
            )

        plt.plot(values, costs)
        plt.vlines(test.S, ymin=np.min(costs), ymax=np.max(costs), label='Current Stock Price',
                   color='black', linestyle='--')
        plt.xlabel('Strike Price/PayOutThreshold E')
        plt.ylabel('Current Cost')
        plt.title('Stock Cost')
        plt.legend()
        plt.show()

    def cash_test():  # pays out 10 dollars at expiration
        test = OptionPrices(r=0.05, S=100, t=0, T=20, sigma=0.1)
        resolution = 30
        costs = np.empty((resolution))
        print(np.shape(costs))
        values = np.linspace(50, 150, resolution)
        for i in tqdm(range(resolution), desc='looping'):

            costs[i] = cost_of_option_to_get_cash_payout_at_T = test.Cash(
                10)

        plt.plot(values, costs)
        plt.vlines(test.S, ymin=np.min(costs), ymax=np.max(costs), label='Current Stock Price',
                   color='black', linestyle='--')
        plt.xlabel('Strike Price/PayOutThreshold E')
        plt.ylabel('Current Cost')
        plt.title('Cash Cost')
        plt.legend()
        plt.show()

    def binary_option_test():
        test = OptionPrices(r=0.05, S=100, t=0, T=1, sigma=0.1)
        resolution = 30
        costs = np.empty((resolution))
        print(np.shape(costs))
        values = np.linspace(50, 150, resolution)
        for i in tqdm(range(resolution), desc='looping'):

            costs[i] = cost_of_option_to_get_cash_payout_at_T = test.BinaryOption(
                values[i])

        plt.plot(values, costs)
        plt.vlines(test.S, ymin=0, ymax=1, label='Current Stock Price',
                   color='black', linestyle='--')
        plt.xlabel('Strike Price/PayOutThreshold E')
        plt.ylabel('Current Cost')
        plt.title('Binary Option Cost')
        plt.legend()
        plt.show()

    def call_option_test():
        test = OptionPrices(r=0.05, S=100, t=0.5, T=1, sigma=0.1)
        resolution = 30
        costs = np.empty((resolution))
        print(np.shape(costs))
        values = np.linspace(50, 150, resolution)
        for i in tqdm(range(resolution), desc='looping'):

            costs[i] = cost_of_option_to_get_cash_payout_at_T = test.CallOption(
                values[i])

        plt.plot(values, costs)
        plt.vlines(test.S, ymin=np.min(costs), ymax=np.max(costs), label='Current Stock Price',
                   color='black', linestyle='--')
        plt.xlabel('Strike Price/PayOutThreshold E')
        plt.ylabel('Current Cost')
        plt.title('Call Option Cost')
        plt.legend()
        plt.show()

    def put_option_test():
        test = OptionPrices(r=0.05, S=100, t=0.5, T=1, sigma=0.1)
        resolution = 30
        costs = np.empty((resolution))
        print(np.shape(costs))
        values = np.linspace(50, 150, resolution)
        for i in tqdm(range(resolution), desc='looping'):

            costs[i] = cost_of_option_to_get_cash_payout_at_T = test.PutOption(
                values[i])

        plt.plot(values, costs)
        plt.vlines(test.S, ymin=np.min(costs), ymax=np.max(costs), label='Current Stock Price',
                   color='black', linestyle='--')
        plt.xlabel('Strike Price/PayOutThreshold E')
        plt.ylabel('Current Cost')
        plt.title('Put Option Cost')
        plt.legend()
        plt.show()


    # binary_option_test()
    # call_option_test()
    # stock_test()
    # cash_test()
    # print(costs)
    # put_option_test()


if __name__ == "__main__":
    main()
