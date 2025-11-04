# (D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice1\venv) D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice2\2-Multiple_Linear_Reg>python 3-.py

from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

with open('model.pkl','rb') as file:
    model = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

@app.route("/", methods=['GET','POST'])
def welcome():
    prediction_text = None  # default value
    if request.method=='POST':
        year = request.form['year']   #2029
        month = request.form['month']   #'January'
        interest_rate = request.form['interest_rate']   #2.75
        unemployment_rate = request.form['unemployment_rate']   #5.3
        df_new = pd.DataFrame({'year':[year],
                        'month':[month],
                        'interest_rate':[interest_rate],
                        'unemployment_rate':[unemployment_rate]})
        df_new.drop(['year','month'], axis = 1, inplace = True)
        df_new = scaler.transform(df_new)
        prediction = model.predict(df_new)
        prediction_text =  f"Prediction about Price of {year} is {prediction[0][0]}"
    return render_template('index.html', prediction_text=prediction_text)

if __name__=="__main__":
    app.run(debug = True)