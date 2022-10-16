import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict():
    Tv=request.args.get('Tv')
    prediction=regmodel.predict(Tv)
    return "The predicted value is"+ (str(prediction))

@app.route('/predict_file',methods=["POST"])
def predict_note():
    df=pd.read_csv(request.files.get("advertising.csv"))
    prediction=regmodel.predict(df)
    return "The predicted values is"+str(list(prediction))



if __name__=="__main__":
    app.run(debug=True)