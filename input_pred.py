import pandas as pd
import joblib
d1=joblib.load("rc1_d.pkl")
w1=joblib.load("rc1_w.pkl")
m1=joblib.load("rc1_m.pkl")
try:
    a=float(input("enter the Open(price):"))
    b=float(input("enter the High(price):"))
    c=float(input("enter the Low(price):"))
    d=float(input("enter the Close(price):"))
    e=float(input("enter the Volume(price):"))
    f=float(input("enter the Day in Date('year-month-'day'):"))
    g=float(input("enter the Month in Date(year-'month'-day):"))
    h=float(input("enter the Year in Date('year'-month-day):"))
    openint=0
except Exception as er:
    print("error\n",er)

df=pd.DataFrame([{
    "Open":a,
    "High":b,
    "Low":c,
    "Close":d,
    "Volume":e,
    "OpenInt":openint,
    "Day":f,
    "Month":g,
    "Year":h
}])
y_pred1=d1.predict(df)[0]
y_pred2=w1.predict(df)[0]
y_pred3=m1.predict(df)[0]

if y_pred1==0:
    y_pred1="Down"
else:
    y_pred1="Up"

if y_pred2==0:
    y_pred2="Down"
else:
    y_pred2="Up"

if y_pred3==0:
    y_pred3="Down"
else:
    y_pred3="Up"


print(f"Stock Or ETF Can Go:\nTomorrow: {y_pred1}\nNext Week: {y_pred2}\nOne month: {y_pred3}")