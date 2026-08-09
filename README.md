# Short-Term Stock Market Prediction

A machine learning project for predicting **short-term stock market direction (Up/Down)** using large-scale historical stock market data containing **350M+ records**.

The project uses historical OHLC data, feature engineering, time-based target creation, and machine learning classification models to predict future market movement over multiple short-term horizons.

> **Disclaimer:** This project is for educational and research purposes only. Stock-market predictions are inherently uncertain and should not be considered financial advice.

---

## Project Overview

The goal of this project is to build a machine learning pipeline that learns patterns from historical stock-market data and predicts whether the future closing price will be higher or lower than the current closing price.

Instead of attempting to predict the exact future stock price, this project treats the problem as a **classification task**:

```text
0 → Down / No Increase
1 → Up / Increase
```

The project supports multiple prediction horizons:

* **1 Day** — next trading day's direction
* **1 Week** — approximately 5 trading days ahead
* **1 Month** — approximately 21 trading days ahead

---

## Dataset

The project works with a large historical stock-market dataset containing **350M+ records**.

Typical market information includes:

* Open
* High
* Low
* Close
* Volume
* Date
* Ticker / Stock Symbol

Large datasets require memory-efficient processing because loading hundreds of millions of rows into memory can exceed available RAM.

---

## Machine Learning Problem

### Input

Historical stock-market information.

### Output

Future market direction:

```text
UP
or
DOWN
```

For example:

```text
Today's Close = ₹100

Future Close = ₹105
→ Target = 1 (UP)

Future Close = ₹95
→ Target = 0 (DOWN)
```

---

# Target Engineering

The future closing price is generated using time shifting.

### 1-Day Target

```python
df["tomorrow_close"] = df["Close"].shift(-1)

df["1d_T"] = (
    df["tomorrow_close"] > df["Close"]
).astype(int)
```

### 1-Week Target

Approximately 5 trading days:

```python
df["1_week_close"] = df["Close"].shift(-5)

df["1w_T"] = (
    df["1_week_close"] > df["Close"]
).astype(int)
```

### 1-Month Target

Approximately 21 trading days:

```python
df["1_month_close"] = df["Close"].shift(-21)

df["1m_T"] = (
    df["1_month_close"] > df["Close"]
).astype(int)
```

---

# Feature Engineering

The project converts raw market data into machine-learning features.

Examples include:

```text
Open
High
Low
Close
Volume
Day
Month
Year
```

Additional technical/time-series features can be created from historical prices.

The date column is converted into a proper datetime format before extracting useful information:

```python
df["Date"] = pd.to_datetime(df["Date"])

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
```

The data is then sorted chronologically.

---

# Avoiding Data Leakage

Time-series prediction is different from normal tabular machine learning.

Randomly mixing past and future observations can cause **data leakage**.

Therefore, the project uses chronological/time-series splitting rather than blindly using a random train-test split.

Conceptually:

```text
Past Data
───────────────
Training Data

        ↓

More Recent Data
───────────────
Testing Data
```

The model should never receive information from the future while learning from the past.

---

# Models

The project experiments with machine-learning models suitable for classification and large datasets.

Examples include:

* Random Forest
* Logistic Regression
* XGBoost
* Decision Trees

The final model can be selected based on:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Generalization on unseen time periods

Accuracy alone should not be used to judge a trading prediction model.

---

# Saved Model Files

The trained machine-learning models can be saved using `joblib`.

Example:

```python
import joblib

joblib.dump(model, "stock_model.pkl")
```

The model can later be loaded without retraining:

```python
model = joblib.load("stock_model.pkl")
```

Possible project files:

```text
models/
│
├── model_1d.pkl
├── model_1w.pkl
└── model_1m.pkl
```

Where:

```text
model_1d.pkl → 1-day prediction
model_1w.pkl → 1-week prediction
model_1m.pkl → 1-month prediction
```

If preprocessing objects are also saved, they can be stored separately:

```text
models/
├── model.pkl
├── scaler.pkl
└── feature_columns.pkl
```

This is important because the exact same preprocessing used during training must be applied during prediction.

---

# Prediction Pipeline

The complete prediction process is:

```text
Historical Market Data
        ↓
Data Cleaning
        ↓
Date Processing
        ↓
Feature Engineering
        ↓
Target Creation
        ↓
Chronological Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Model (.pkl)
        ↓
Load Model
        ↓
New Market Data
        ↓
Prediction
        ↓
UP / DOWN
```

---

# Loading the Model

Example:

```python
import joblib

model = joblib.load("models/model_1d.pkl")
```

Then prepare the input features:

```python
X_new = df[features]
```

Prediction:

```python
prediction = model.predict(X_new)
```

Probability:

```python
probability = model.predict_proba(X_new)
```

Example output:

```text
Prediction: UP

Probability:
DOWN → 0.32
UP   → 0.68
```

The probability represents the model's estimated class probability, **not the probability that the stock will actually rise**.

---

# Project Structure

```text
Stock-Market-Prediction/
│
├── data/
│   └── ...
│
├── models/
│   ├── model_1d.pkl
│   ├── model_1w.pkl
│   └── model_1m.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/aaradhya-gavali13/YOUR-REPOSITORY.git
```

Move into the project:

```bash
cd YOUR-REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
joblib
matplotlib
xgboost
```

Install them with:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the training pipeline:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

Or, if the project uses a main file:

```bash
python main.py
```

---

# Git and Large Files

Because the original dataset contains **350M+ records**, it may be too large to store directly on GitHub.

Large datasets should generally be excluded:

```gitignore
data/
*.csv
*.parquet
```

Similarly, large model files may need separate storage:

```gitignore
*.pkl
*.joblib
```

However, if your `.pkl` models are small enough and you intentionally want users to download them from the repository, you can track them.

Do **not** upload:

```text
.env
API keys
passwords
private credentials
```

Use `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# Evaluation

The classification model should be evaluated using multiple metrics.

### Accuracy

Percentage of correct predictions.

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

### Precision

Measures how many predicted positive movements were actually positive.

### Recall

Measures how many actual positive movements were detected.

### F1 Score

Balances precision and recall.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Confusion matrix:

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
```

---

# Important Considerations

A high test score does **not automatically mean the model is profitable**.

Stock-market prediction has several challenges:

* Market regime changes
* Noise
* Transaction costs
* Slippage
* Brokerage charges
* Liquidity
* Unexpected news
* Corporate events
* Overfitting
* Data leakage

A model that performs well on historical data may perform poorly on future market conditions.

Therefore, a proper trading-system evaluation should eventually include:

```text
Model Prediction
       ↓
Trading Strategy
       ↓
Transaction Costs
       ↓
Position Sizing
       ↓
Backtesting
       ↓
Drawdown
       ↓
Risk-adjusted Returns
```

---

# Future Improvements

Possible improvements include:

* Technical indicators
* Rolling returns
* Moving averages
* RSI
* MACD
* Volatility features
* Market-index features
* Sector information
* Walk-forward validation
* Hyperparameter optimization
* Feature importance analysis
* Backtesting
* Transaction-cost simulation
* Portfolio-level prediction
* Real-time prediction API

---

# Technologies Used

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Programming           |
| Pandas       | Data processing       |
| NumPy        | Numerical computation |
| Scikit-learn | Machine learning      |
| XGBoost      | Gradient boosting     |
| Joblib       | Model serialization   |
| Matplotlib   | Visualization         |
| Git/GitHub   | Version control       |

---

# Author

**Aaradhya Gavali**

GitHub: `aaradhya-gavali13`

---

## Disclaimer

This project is an educational machine-learning experiment and is not financial advice. Predictions generated by the model should not be interpreted as guaranteed future market movements or investment recommendations.
