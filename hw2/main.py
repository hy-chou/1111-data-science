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

print('The optimal projection vector is ', w_v[0], ',')
print('normalized to unit length, ', w_v[0] / np.linalg.norm(w_v[0]), '.')
print('The corresponding eigenvalue is ', w_lambda[0], '.')
