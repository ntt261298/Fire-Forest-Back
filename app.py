from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.externals import joblib
from pipeline import AttributeSelector, CustomBinarizer, FullPipeline

# instantiate flask
app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb://ntt261298:hust123456@ds235401.mlab.com:35401/forest-fire"
mongo = PyMongo(app)

CORS(app)

model = joblib.load('./models/sgd_model.pkl')

@app.route('/predict', methods=["POST"])
def predict():
    result = {'success': False}
    data = request.json
    print(request.json)
    if(data is None):
        return jsonify({'message': 'Data is null'})
    print(data)

    x = pd.DataFrame([data], columns=data.keys())    
    pipeline = FullPipeline()
    data_prepared = pipeline.prepare_data(x)
    print(data_prepared)
    
    result["prediction"] = str(model.predict(data_prepared))
    result["success"] = True

    # return a reponse in json format
    return jsonify(result)

@app.route('/data', methods=["POST"])
def data():
    result = {'success': False}
    data = request.json
    print(request.json)
    if(data is None):
        return jsonify({'message': 'Data is null'})
    print(data)

    mongo.db.dataset.save(data)

    result["success"] = True

    # return a reponse in json format
    return jsonify(result)
    

app.run(host='0.0.0.0', debug=True)
