


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
og=pd.read_csv(r"C:\Users\PASKAL\OneDrive\Desktop\python_aaradhya\Huge_stock_market\predict.csv")
df=pd.DataFrame(og)
# Data Cleaning
df = df.head(100000)     # 2 million rows
df["Date"]=pd.to_datetime(df["Date"],format="%Y-%m-%d")
df["Day"]=df["Date"].dt.day
df["Month"]=df["Date"].dt.month
df["Year"]=df["Date"].dt.year
df.sort_values(by=["Date"],inplace=True)
df.drop(["Date","File"],axis=1,inplace=True)
df["tomorrow_close"]=df["Close"].shift(-1)
df["1d_T"]=(df["tomorrow_close"]>df["Close"]).astype(int)
df["1_week_close"]=df['Close'].shift(-5)
df["1w_T"]=(df["1_week_close"]>df["Close"]).astype(int)
df["1_month_close"]=df["Close"].shift(-21)
df["1m_T"]=(df["1_month_close"]>df["Close"]).astype(int)
df.dropna(inplace=True)

df.drop(["tomorrow_close","1_week_close","1_month_close"],axis=1,inplace=True)
x=df.drop(["1d_T","1w_T","1m_T"],axis=1)
d=df["1d_T"]
w=df["1w_T"]
m=df["1m_T"]
time_series=TimeSeriesSplit(n_splits=10)
for x_t,x_te in time_series.split(x):
    pass
x_train=x.iloc[x_t]
x_test=x.iloc[x_te]
d_train=d.iloc[x_t]
d_test=d.iloc[x_te]
w_train=w.iloc[x_t]
w_test=w.iloc[x_te]
m_train=m.iloc[x_t]
m_test=m.iloc[x_te]

rc=RandomForestClassifier(random_state=42)
rc.fit(x_train,m_train)
y_pred=rc.predict(x_test)

print(x.columns)
a = 120.5      # Open
b = 123.0      # High
c = 119.8      # Low
d = 122.1      # Close
e = 1500000    # Volume
f = 15         # Day
g = 6          # Month
h = 2026       # Year
openint = 0    # OpenInt
145.2      # Open
b = 148.7      # High
c = 143.9      # Low
d = 147.5      # Close
e = 2350000    # Volume
f = 22         # Day
g = 3          # Month
h = 2026       # Year
openint = 0    # OpenInt

df = pd.DataFrame({
    "Open": [a],
    "High": [b],
    "Low": [c],
    "Close": [d],
    "Volume": [e],
    "Day": [f],
    "Month": [g],
    "Year": [h],
    "OpenInt": [openint]
})

