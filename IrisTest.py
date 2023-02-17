import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from NeuralNetwork import NeuralNetwork
from Layer import Layer, InputLayer

data = pd.read_csv("Iris/Iris.csv", index_col="Id")

data['Species'] = data['Species'].map({'Iris-setosa': [1, 0, 0],
                                       'Iris-versicolor': [0, 1, 0],
                                       'Iris-virginica': [0, 0, 1]})

input = InputLayer(4)
hidden1 = Layer(4, 3)
output = Layer(3, 3)

nn = NeuralNetwork([input, hidden1, output], 0.001)

X = data[['SepalLengthCm', 'SepalWidthCm',
          'PetalLengthCm', 'PetalWidthCm']].to_numpy()
Y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.6)

nn.train(X_train, y_train, 10000)

answer = []
for i in X_test:
    answer.append(nn.predict(i))

answer = np.array(answer)

print("Accuracy:", accuracy_score(y_test.map(np.argmax), answer.argmax(axis=1)))
target_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
print(classification_report(y_test.map(np.argmax), answer.argmax(axis=1), target_names=target_names))
print(confusion_matrix(y_test.map(np.argmax), answer.argmax(axis=1)))
