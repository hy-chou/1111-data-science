from math import exp
from random import random, randrange, sample, seed

from matplotlib import pyplot
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

seed(1115)


x = read_csv('hw3_Data1/gene.txt', delimiter=' ', header=None).to_numpy().T
y = read_csv('hw3_Data1/label.txt', header=None).to_numpy()
y = (y > 0).astype(int).reshape(y.shape[0])


init_temp = 10000
final_temp = 0
k = 1e-5
init_state = set(sample(range(2000), k=randrange(1, 2001)))
estimator = DecisionTreeClassifier(random_state=1115)
score_history = []
n_feat_history = []


this_state = init_state
n_feat_history.append(len(this_state))

this_x = x[:, list(this_state)]
this_score = cross_val_score(estimator, this_x, y).mean()
score_history.append(this_score)

for temp in range(init_temp, final_temp, -1):
    next_state = this_state ^ set([randrange(2000)])
    n_feat_history.append(len(next_state))

    next_x = x[:, list(next_state)]
    next_score = cross_val_score(estimator, next_x, y).mean()
    score_history.append(next_score)

    score_difference = next_score - this_score

    if score_difference >= 0:
        this_state = next_state.copy()
        this_score = next_score
    else:
        prob = exp(score_difference / temp / k)
        if random() < prob:
            this_state = next_state.copy()
            this_score = next_score


print(f'At the {argmax(score_history)}-th iteration,')
print(f'Best score:  {max(score_history)}')
print(f'Number of features:  {n_feat_history[argmax(score_history)]}')


fig, ax = pyplot.subplots()

ax.plot(range(len(score_history)), score_history, label='decision tree')
ax.set_xlabel('Iteration')
ax.set_ylabel('CV score')
ax.set_title('Problem 2')
ax.legend()

pyplot.savefig('hw3-p2.png', bbox_inches='tight')
pyplot.close(fig)
