import numpy as np
import pandas as pd
import time
import json

from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import datasets


class SVM:

    targets_testes = []
    base = 0

    def __init__(self, obj):
        print("SVM iniciado")
        self.targets_testes = []
        self.base = 0
        json_received = obj
        self.base = json_received['base']
        if self.base == 1:
            self.targets_testes.append(json_received['parametro1'])
            self.targets_testes.append(json_received['parametro2'])
            self.targets_testes.append(json_received['parametro3'])
            self.targets_testes.append(json_received['parametro4'])
        else:
            self.targets_testes.append(json_received['parametro1'])
            self.targets_testes.append(json_received['parametro2'])
            self.targets_testes.append(json_received['parametro3'])
            self.targets_testes.append(json_received['parametro4'])
            self.targets_testes.append(json_received['parametro5'])
            self.targets_testes.append(json_received['parametro6'])
            self.targets_testes.append(json_received['parametro7'])
            self.targets_testes.append(json_received['parametro8'])
            self.targets_testes.append(json_received['parametro9'])
            self.targets_testes.append(json_received['parametro10'])
            print("segunda base")

    def load_iris_data(self):
        if self.base == 1:
            data = pd.read_csv("IRIS.csv")
            features = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
            target_variables = data.Class
            json_object = (self.generate_graphics(features.as_matrix(), target_variables.as_matrix()))
            return self.train_data(json_object, features.as_matrix(), target_variables.as_matrix())
        else:
            data = pd.read_csv("POKER-HAND.csv")
            features = data[["SuitOfCard1", "RankOfCard1", "SuitOfCard2", "RankOfCard2", "SuitOfCard3", "RankOfCard3", "SuitOfCard4", "RankOfCard4", "SuitOfCard5", "RankOfCard5"]]
            target_variables = data.Class
            # json_object = (self.generate_graphics(features.as_matrix(), target_variables.as_matrix()))
            json_object=[{
                "teste":1
            }]
            return self.train_data(json_object, features.as_matrix(), target_variables.as_matrix())

    def train_data(self, json_object, features, target_variables):
        model = svm.SVC(gamma=0.001, C=100.0, probability=True, kernel='linear')
        t0 = time.clock()
        print(features)
        fitted_model = model.fit(features, target_variables)
        t1 = time.clock()
        training_time = t1 - t0
        t2 = time.clock()

        if self.base == 1:
            predictions = fitted_model.predict(np.array(self.targets_testes).reshape(-1, 4))
        else:
            predictions = fitted_model.predict(np.array(self.targets_testes).reshape(-1, 10))
        t3 = time.clock()
        prediction_time = t3 - t2

        if self.base == 1:
            return self.return_data(json_object,
                                    fitted_model.predict_proba(np.array(self.targets_testes).reshape(-1, 4)),
                                    predictions, training_time, prediction_time)
        else:
            return self.return_data(json_object,
                                    fitted_model.predict_proba(np.array(self.targets_testes).reshape(-1, 10)),
                                    predictions, training_time, prediction_time)

    @staticmethod
    def generate_graphics(features, target_variables):
        n_classes = 3
        plot_colors = "ryb"
        plot_step = 0.02
        iris = datasets.load_iris()
        data = {}
        contador = 0
        for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                        [1, 2], [1, 3], [2, 3]]):

            X = features[:, pair]
            y = iris.target

            clf = svm.SVC(gamma=0.001, C=100.0, kernel='linear').fit(X, y)

            # Plot the decision boundary
            plt.subplot(2, 3, pairidx + 1)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

            plt.xlabel(iris.feature_names[pair[0]])
            plt.ylabel(iris.feature_names[pair[1]])

            # Plot the training points
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

                data['X' + str(contador)] = X[idx, 0].tolist()
                data['Y' + str(contador)] = X[idx, 1].tolist()
                data['color' + str(contador)] = color
                contador += 1

        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        json_data = json.dumps(data)
        return json_data

    @staticmethod
    def return_data(json_object, predict_proba, predictions, training_time, prediction_time):
        # confusion_matrix_return = confusion_matrix(target_test, predictions)
        # accuracy_return = accuracy_score(target_test, predictions)
        return json.dumps({
            'tipo': 'SVM',
            'grafico': json_object,
            'prediction': predictions.tolist(),
            'accuracy': predict_proba.tolist(),
            'training_time': training_time,
            'prediction_time': prediction_time
        })


