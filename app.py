from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.externals import joblib
from pipeline import AttributeSelector, CustomBinarizer, FullPipeline

# instantiate flask
app = Flask(__name__)
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


app.run(host='0.0.0.0', debug=True)
