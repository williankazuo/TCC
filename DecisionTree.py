import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import datasets
from sklearn import metrics


class DecisionTree:

    targets_testes = []
    base = 0

    def __init__(self, obj):
        print("DecisionTree iniciado")
        self.targets_testes = []
        self.base = 0
        # json_received = json.loads(obj)
        json_received = obj
        self.base = json_received['base']
        if self.base == 1:
            self.targets_testes.append(json_received['parametro1'])
            self.targets_testes.append(json_received['parametro2'])
            self.targets_testes.append(json_received['parametro3'])
            self.targets_testes.append(json_received['parametro4'])
        else:
            print('segunda base')

    def load_iris_data(self):
        if self.base == 1:
            data = pd.read_csv("IRIS.csv")
            features = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
            target_variables = data.Class
            json_object = (self.generate_graphics(features.as_matrix(), target_variables.as_matrix()))

            return self.train_data(json_object, features.as_matrix(), target_variables.as_matrix())
        # return "Teste"

    def train_data(self, json_object, features, target_variables):
        model = DecisionTreeClassifier()
        t0 = time.clock()
        fitted_model = model.fit(features, target_variables)
        t1 = time.clock()
        training_time = t1 - t0

        self.generate_graph_viz(fitted_model)

        t2 = time.clock()
        predictions = fitted_model.predict(np.array(self.targets_testes).reshape(-1, 4))
        t3 = time.clock()
        prediction_time = t3 - t2

        # print(fitted_model.predict_proba(np.array(self.targets_testes).reshape(-1, 4)))
        return self.return_data(json_object, fitted_model.predict_proba(np.array(self.targets_testes).reshape(-1, 4)), predictions, training_time, prediction_time)

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

            clf = DecisionTreeClassifier().fit(X, y)

            #Plot the decision boundary
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

            #Plot the training points
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

                data['X'+str(contador)] = X[idx, 0].tolist()
                data['Y'+str(contador)] = X[idx, 1].tolist()
                data['color'+str(contador)] = color
                contador += 1
        json_data = json.dumps(data)
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        return json_data


    @staticmethod
    def generate_graph_viz(fitted_model):
        tree.export_graphviz(fitted_model, out_file='tree.dot')

    @staticmethod
    def return_data(json_object, predict_proba, predictions, training_time, prediction_time):
        # confusion_matrix_return = confusion_matrix(a, predictions)
        # accuracy_return = accuracy_score(a, predictions)
        return json.dumps({
            'tipo': 'Decision Tree',
            'grafico': json_object,
            'prediction': predictions.tolist(),
            'accuracy': predict_proba.tolist(),
            'training_time': training_time,
            'prediction_time': prediction_time
        })



