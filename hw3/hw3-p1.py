from matplotlib import pyplot
from numpy import argmax, argsort, negative
from pandas import read_csv
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


x = read_csv('hw3_Data1/gene.txt', delimiter=' ', header=None).to_numpy().T
y = read_csv('hw3_Data1/label.txt', header=None).to_numpy()
y = (y > 0).astype(int).reshape(y.shape[0])


mutual_info = mutual_info_classif(x, y)
ranking_idx = argsort(negative(mutual_info))


svc = SVC(random_state=1115)
score_history = []
for m in range(1, 2001):
    x_selected = x[:, ranking_idx[:m]]
    scores = cross_val_score(svc, x_selected, y)
    score_history.append(scores.mean())


print(f'Best score:  {max(score_history)}')
print(f'Number of features:  {1 + argmax(score_history)}')


fig, ax = pyplot.subplots()

ax.plot(range(1, 2001), score_history, label='SVC')
ax.set_xlabel('Number of features selected')
ax.set_ylabel('CV score')
ax.set_title('Problem 1')
ax.legend()

pyplot.savefig('hw3-p1.png', bbox_inches='tight')
pyplot.close(fig)
