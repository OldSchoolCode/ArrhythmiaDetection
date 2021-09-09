#from load import *
import random as rand
#from random import seed
from keras.models import load_model
import flask
from flask import Flask, jsonify, request
import json
import pickle
import joblib
import sys
import os
import flask
from model import *

app = Flask(__name__)

# Load the pre-trained/pickled model 
model = load_model('model.h5')
###################################################

@app.route('/')
def index():
    return flask.render_template('index.html')


# seed the pseudorandom number generator
#from random import random
# seed random number generator
#seed(1)

@app.route('/predict', methods=['POST'])
def predict():
	
# Render output as html using predict.html
	if model.prediction >= threshold:
		model.prediction = "Positive"
	else:
		model.prediction = "Negative"
		return flask.render_template('predict.html', prediction=prediction)



