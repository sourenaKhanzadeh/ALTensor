from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

x = np.array([i for i in range(2000)])
y = []

for i in x:
    if i % 2 == 0:
        y.append(1)
    else:
        y.append(0)

y = np.array(y)

x = x.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(max_iter=10000)
lr.fit(x_train, y_train)

print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))

print(lr.predict(np.array([i for i in range(0, 2000, 1)]).reshape(-1, 1)))