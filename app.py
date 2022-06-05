import pickle
from flask import Flask, request, app, jsonify, Response, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model_svr.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])         # creates an api
def predict_api():
    """ DIRECT API CALLS THROUGH REQUEST"""
    data = request.json["data"]                      # reads the info from postman, a dictionary key
    print(data)
    new_data = [list(data.values())]                 # creates a 2d array from the values of the dict
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route("/predict",methods=["POST"])               # creates an api
def predict():
    """ DIRECT API CALLS THROUGH REQUEST"""
    data = [float(x) for x in request.form.values()]                      # reads all the values
    print(data)
    final_features = [np.array(data)]
    #print(final_features)
    #print(model.predict(final_features))
    output = model.predict(final_features)[0]
    #print(output)
    return render_template("home.html", prediction_text="Airfoil pressure is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)