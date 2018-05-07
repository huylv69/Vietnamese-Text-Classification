from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

max_iter = [20, 50, 70, 100, 200]
penalty =['l1', 'l2']
Cs=[0.1, 10, 20, 30]
X_axis=["20:l1", "20:l2", "50:l1", "50:l2", "70:l1","70:l2","100:l1","100:l2","200:l1","200:l2"]
X=[1,2,3,4,5,6,7,8,9,10]
scores = np.load('drawLogisChart.npy')
print(scores)

for ind, i in enumerate(Cs):
	plt.plot(X, scores[ind], label='C: ' + str(i))

plt.legend()
plt.title("Logistic Regesstion", fontweight='bold')
plt.xlabel('max_iter and penalty')
# plt.text(1.5,0.92,"logictic: 0.9230723658876152, C: 30, max_iter: 20, penalty: 'l2'")
plt.text(1.5,0.925,"C:30, max_iter:20, penalty:'l2'")
plt.ylabel('Mean score')
plt.grid('on')
plt.xticks(X, X_axis)
plt.xlim(1,10)
plt.legend(loc="best", fontsize=15)
plt.show()
