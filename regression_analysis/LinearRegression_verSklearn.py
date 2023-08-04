import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston.data[:, 5:6]  # RM: average number of rooms per dwelling
y = boston.target

reg = LinearRegression()
reg.fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, reg.predict(X), color='red', linewidth=2)
plt.show()

print("回帰係数:", reg.coef_[0])
print("切片:", reg.intercept_)
print("評価:", reg.score(X, y))
