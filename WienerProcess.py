# conda install -n environment package
# conda info --env
# conda create -name new_environment
# conda list -n environment

import numpy as np
import matplotlib.pyplot as plt
import sympy
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
