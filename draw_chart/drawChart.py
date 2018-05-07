from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

max_iter = [20, 50, 70]
penalty =['l2']
Cs=[0.1,1,10,20]
X=[1,3,5,7,10]
X_axis=["1:l2","10:l2", "20:l2", "50:l2","100:l2"]
scores = np.load('drawChart.npy')
print(scores)


for ind, i in enumerate(Cs):
	plt.plot(X, scores[ind], label='C: ' + str(i))

plt.legend()
plt.title("LinearSVC", fontweight='bold')
plt.xlabel('max_iter and penalty')
plt.ylabel('Mean score')
plt.grid('on')
plt.xticks(X, X_axis)
plt.xlim(1,10)
plt.legend(loc="best", fontsize=15)
plt.show()
