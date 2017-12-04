import numpy as np
import pandas as pd
import time
import json

from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import naive_bayes


class NaiveBayes:

    def __init__(self):
        print("NaiveBayes iniciado")

    def load_iris_data(self):
        data = pd.read_csv("IRIS.csv")
        features = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
        target_variables = data.Class
        return self.train_data(features, target_variables)

    def train_data(self, features, target_variables):
        # 80% do dataset está sendo usado para treino(variaveis train), e 20% está sendo usado para testes(variaveis Tes).
        feature_train, feature_test, target_train, target_test = train_test_split(features, target_variables, test_size=.2)

        model = naive_bayes.GaussianNB()
        t0 = time.clock()
        fitted_model = model.fit(feature_train, target_train)
        t1 = time.clock()
        training_time = t1 - t0

        t2 = time.clock()
        predictions = fitted_model.predict(feature_test)
        t3 = time.clock()
        prediction_time = t3 - t2

        return self.return_data(target_test, predictions, training_time, prediction_time)

    @staticmethod
    def return_data(target_test, predictions, training_time, prediction_time):
        confusion_matrix_return = confusion_matrix(target_test, predictions)
        accuracy_return = accuracy_score(target_test, predictions)
        return json.dumps({
            'tipo': 'Naive Bayes',
            'confusion_matrix': confusion_matrix_return.tolist(),
            'accuracy': accuracy_return,
            'training_time': training_time,
            'prediction_time': prediction_time
        })


