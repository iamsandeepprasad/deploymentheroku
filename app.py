import numpy as np
import pandas as pd
import pickle
model=pickle.load(open("model.pkl","rb"))
from flask import Flask,request,render_template
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["post"])
def predictiom():
    int_feature=[int(x) for x in request.form.values()]
    final_feature=[np.array(int_feature)]
    prediction=model.predict(final_feature)
    output=prediction
    return render_template('index.html',prediction_text=output)

if __name__=="__main__":
    app.run(debug=True)
