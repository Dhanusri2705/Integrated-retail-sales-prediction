import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import scipy.stats as stats


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")
st.title("📊 Universal Retail Analytics Dashboard")

# ------------------------------------------------
# SIDEBAR MENU
# ------------------------------------------------
option = st.sidebar.selectbox("Select Analysis", [
    "Overview",
    "Index Numbers",
    "Regression",
    "ANOVA & Tests",
    "Time Series",
    "Feature Selection",
    "Performance Metrics"
])

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------------------------
    # COLUMN DETECTION
    # ------------------------------------------------
    date_col = None
    sales_col = None
    quantity_col = None
    customer_col = None
    product_col = None

    for col in df.columns:
        c = col.lower()

        if "date" in c:
            date_col = col
        if "sales" in c or "price" in c or "amount" in c:
            sales_col = col
        if "quantity" in c:
            quantity_col = col
        if "customer" in c:
            customer_col = col
        if "product" in c or "description" in c:
            product_col = col

    # ------------------------------------------------
    # SALES CALCULATION
    # ------------------------------------------------
    if quantity_col and sales_col:
        df["TotalSales"] = df[quantity_col] * df[sales_col]
    elif sales_col:
        df["TotalSales"] = df[sales_col]
    else:
        st.error("No sales column found")
        st.stop()

    # ------------------------------------------------
    # DATE PROCESSING
    # ------------------------------------------------
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df["YearMonth"] = df[date_col].dt.to_period("M")

        sales_month = df.groupby("YearMonth")["TotalSales"].sum()
        sales_month.index = sales_month.index.to_timestamp()

    # ------------------------------------------------
    # OVERVIEW
    # ------------------------------------------------
    if option == "Overview":

        st.subheader("Key Metrics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Revenue", f"{df['TotalSales'].sum():,.0f}")
        c2.metric("Transactions", len(df))
        c3.metric("Customers", df[customer_col].nunique() if customer_col else 0)
        c4.metric("Avg Order", f"{df['TotalSales'].mean():.2f}")

        st.subheader("Descriptive Statistics")
        numeric = df.select_dtypes(include="number")
        st.dataframe(numeric.describe())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ------------------------------------------------
    # INDEX NUMBERS
    # ------------------------------------------------
    if option == "Index Numbers":

        base = df.iloc[0]["TotalSales"]

        simple = df["TotalSales"].sum() / base * 100
        weighted = (df["TotalSales"] * df["TotalSales"]).sum() / df["TotalSales"].sum()

        laspeyres = simple
        paasche = weighted
        fisher = (laspeyres * paasche) ** 0.5

        st.write("Simple Index:", simple)
        st.write("Weighted Index:", weighted)
        st.write("Laspeyres:", laspeyres)
        st.write("Paasche:", paasche)
        st.write("Fisher Index:", fisher)

    # ------------------------------------------------
    # REGRESSION
    # ------------------------------------------------
    if option == "Regression":

        numeric = df.select_dtypes(include="number").dropna()

        if "TotalSales" in numeric.columns and len(numeric.columns) > 1:

            X = numeric.drop(columns=["TotalSales"])
            y = numeric["TotalSales"]

            model = LinearRegression()
            model.fit(X, y)

            st.write("Linear Regression Coefficients:", model.coef_)

            poly = PolynomialFeatures(2)
            X_poly = poly.fit_transform(X)

            model.fit(X_poly, y)
            st.write("Polynomial Regression Done")

            y_binary = (y > y.mean()).astype(int)

            log_model = LogisticRegression()
            log_model.fit(X, y_binary)

            st.write("Logistic Regression Done")

        else:
            st.warning("Not enough numeric data")

    # ------------------------------------------------
    # ANOVA & TESTS
    # ------------------------------------------------
    if option == "ANOVA & Tests":

        numeric = df.select_dtypes(include="number")

        if len(numeric.columns) >= 2:

            f, p = stats.f_oneway(numeric.iloc[:,0], numeric.iloc[:,1])
            st.write("One-Way ANOVA p-value:", p)

            u, p = stats.mannwhitneyu(numeric.iloc[:,0], numeric.iloc[:,1])
            st.write("Mann-Whitney p-value:", p)

    # ------------------------------------------------
    # TIME SERIES
    # ------------------------------------------------
    if option == "Time Series":

        if date_col:

            result = adfuller(sales_month)
            st.write("ADF Test p-value:", result[1])

            ma = sales_month.rolling(3).mean()
            exp = sales_month.ewm(span=3).mean()

            model = ARIMA(sales_month, order=(1,1,1))
            fit = model.fit()
            forecast = fit.forecast(3)

            fig, ax = plt.subplots()
            ax.plot(sales_month, label="Original")
            ax.plot(ma, label="MA")
            ax.plot(exp, label="Exp")
            ax.plot(forecast, label="Forecast")

            ax.legend()
            st.pyplot(fig)

    # ------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------
    if option == "Feature Selection":

        numeric = df.select_dtypes(include="number").dropna()

        if "TotalSales" in numeric.columns:

            X = numeric.drop(columns=["TotalSales"])
            y = numeric["TotalSales"]

            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()
            st.text(model.summary())

    # ------------------------------------------------
    # PERFORMANCE METRICS
    # ------------------------------------------------
    if option == "Performance Metrics":

        if date_col:

            model = ARIMA(sales_month, order=(1,1,1)).fit()
            forecast = model.forecast(3)

            y_true = sales_month[-3:]

            mse = mean_squared_error(y_true, forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, forecast)

            st.write("MSE:", mse)
            st.write("RMSE:", rmse)
            st.write("MAE:", mae)
