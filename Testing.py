#!flask/bin/python
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from DecisionTree import DecisionTree
from SVM import SVM
from NaiveBayes import NaiveBayes

import json

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/getdecisiontree', methods=['POST'])
def get_decision_tree():
    obj = request.json
    tree = DecisionTree(obj)
    return jsonify(tree.load_iris_data())


@app.route('/api/getsvm', methods=['POST'])
def get_svm():
    obj = request.json
    s = SVM(obj)
    return jsonify(s.load_iris_data())


@app.route('/api/getnaive', methods=['POST'])
def get_naive():
    obj = request.json
    naive = NaiveBayes(obj)
    return jsonify(naive.load_iris_data())


@app.route('/Application/page')
def main_page():
    return send_from_directory('Application', 'Main.html')


@app.route('/Application/Javascript/jquery-3.2.1.min.js')
def main_js():
    return send_from_directory('Application/Javascript', 'jquery-3.2.1.min.js')


@app.route('/Application/CSS/style.css')
def main_css():
    return send_from_directory('Application/CSS', 'style.css')


if __name__ == '__main__':
    app.run(debug=True)
