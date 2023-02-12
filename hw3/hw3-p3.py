from matplotlib import pyplot
from numpy import square
from pandas import read_csv
from pmdarima.arima import ARIMA


train = read_csv('hw3_Data2/train.csv', delimiter=',', usecols=[4])
train = train.to_numpy().squeeze()
test = read_csv('hw3_Data2/test.csv', delimiter=',', usecols=[4])
test = test.to_numpy().squeeze()


order = (0, 1, 0)
seasonal_order = (0, 0, 0, 0)
forecast = ARIMA(order, seasonal_order).fit_predict(train, n_periods=len(test))


print(square(test - forecast).mean())


fig, ax = pyplot.subplots()

ax.plot(range(225), train, label='train')
ax.plot(range(225, 246), test, label='test')
ax.plot(range(225, 246), forecast, dashes=[2, 2],  label='forecast')
ax.set_title(f'(p, d, q), (P, D, Q, s) = {order}, {seasonal_order}')
ax.legend()

pyplot.savefig(f'hw3-p3.png', bbox_inches='tight')
pyplot.close(fig)
