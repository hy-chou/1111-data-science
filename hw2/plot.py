import matplotlib.pyplot as plt
import numpy as np


# data
omega1 = np.array([[5, 3], [3, 5], [3, 4], [4, 5], [4, 7], [5, 6]])
omega2 = np.array([[9, 10], [7, 7], [8, 5], [8, 8], [7, 2], [10, 8]])
x = np.concatenate((omega1, omega2), axis=0)
n1 = omega1.shape[0]
n2 = omega2.shape[0]

# mean
mu1 = np.mean(omega1, axis=0)
mu2 = np.mean(omega2, axis=0)
mu = np.mean(x, axis=0)

# between-class variance
sb1 = np.matmul(np.expand_dims(mu1 - mu, 1),
                np.expand_dims(mu1 - mu, 1).T)
sb2 = np.matmul(np.expand_dims(mu2 - mu, 1),
                np.expand_dims(mu2 - mu, 1).T)
sb = n1 * sb1 + n2 * sb2

# mean-centering data
d1 = omega1 - mu1
d2 = omega2 - mu2

# within-class variance
sw1 = np.matmul(d1.T, d1)
sw2 = np.matmul(d2.T, d2)
sw = sw1 + sw2

# transformation matrix
w = np.matmul(np.linalg.inv(sw), sb)
w_lambda, w_v = np.linalg.eig(w)

# transformed data
omega1v1 = np.matmul(omega1, w_v[0])
omega2v1 = np.matmul(omega2, w_v[0])


# plot
fig, (ax1, ax2) = plt.subplots(1, 2)

# original data
ax1.plot(omega1[:, 0], omega1[:, 1], 'r.', label='class1')
ax1.plot(omega2[:, 0], omega2[:, 1], 'b.', label='class2')
ax1.plot(mu1[0], mu1[1], 'rx',  label='mu1')
ax1.plot(mu2[0], mu2[1], 'bx',  label='mu2')
ax1.plot(mu[0], mu[1], 'kx',  label='mu')
ax1.legend()

# projected data
ax2.plot(omega1v1, np.zeros(n1),  'r.', label='class1')
ax2.plot(omega2v1, np.zeros(n2),  'b.', label='class2')
print(omega1v1)
print(omega2v1)


plt.show()
