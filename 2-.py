import streamlit as st
import pickle
import pandas as pd

st.title("📈 Price Index Prediction 💸💰")
st.text("Welcome!! 🤝 Are you interested to know Price for future year? You are at the right place.")

with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

year = st.text_input("Enter Year in number") #2029
month = st.text_input("Enter Month in number") #'January'
interest_rate = st.text_input("Enter the interest rate in number") #2.75
unemployment_rate = st.text_input("Enter Unemployment rate in number") #5.3

if st.button("Submit") and year and month and interest_rate and unemployment_rate:

    year = int(year) #2029
    month = int(month) #'January'
    interest_rate = float(interest_rate) #2.75
    unemployment_rate = float(unemployment_rate) #5.3

    new_df = pd.DataFrame({'year' : [year],
                        'month' : [month],
                        'interest_rate':[interest_rate],
                        'unemployment_rate':[unemployment_rate]})

    new_df.drop(['year','month'], axis = 1, inplace = True)
    new_df = scaler.transform(new_df)
    new_df_pred = model.predict(new_df) # 1464
    st.success(f"Predicted price is: {new_df_pred[0][0]:.2f}")
