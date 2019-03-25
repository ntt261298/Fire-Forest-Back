from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
keras.__version__
from keras.models import load_model
from flask_cors import CORS, cross_origin

# instantiate flask
app = Flask(__name__)
CORS(app)

# we need to redefine our loss function in order to use it when loading the model
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom loss function
global graph
graph = tf.get_default_graph()
model = load_model('forest-fire.h5', custom_objects={'auc': auc})

@app.route('/predict', methods=["POST"])
def predict():
    result = {'success': False}
    data = request.form
    if(data is None):
        return jsonify({'message': 'Data is null'})
    
    x = pd.DataFrame.from_dict(data, orient='index').transpose()
    with graph.as_default():
        # a = model.predict(x)
        result["prediction"] = str(model.predict(x, batch_size=1,verbose=1))
        result["success"] = True

    # return a reponse in json format
    return jsonify(result)


app.run(host='0.0.0.0', debug=True)
