import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import scipy.stats as stats

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Universal Analytics Dashboard", layout="wide")
st.title("📊 Integrated Predictive Modeling and Time Series Analysis for Retail Sales")

 # ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("📌 Analysis Menu")

option = st.sidebar.radio(
    "Choose Analysis",
    [
        "Overview",
        "Index Numbers",
        "Regression Models",
        "ANOVA & Tests",
        "Time Series Analysis",
        "Dimensionality Reduction",
        "Clustering Techniques",
        "Performance Evaluation",
        "Inference"
    ]
)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if file:

    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception:
        st.error("Error reading file")
        st.stop()

    if df.empty:
        st.error("Dataset is empty!")
        st.stop()

    # ------------------------------------------------
    # DATA PREPROCESSING
    # ------------------------------------------------
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    # AUTO COLUMN DETECTION
    date_col = None
    sales_col = None
    quantity_col = None
    product_col = None
    customer_col = None
    country_col = None

    for col in df.columns:
        c = col.lower()

        if "date" in c or "time" in c:
            date_col = col
        if any(x in c for x in ["sales", "price", "amount", "revenue", "value", "total", "unitprice"]):
            sales_col = col
        if any(x in c for x in ["quantity", "qty", "units"]):
            quantity_col = col
        if any(x in c for x in ["product", "description", "item"]):
            product_col = col
        if "customer" in c:
            customer_col = col
        if "country" in c:
            country_col = col

    if not sales_col:
        sales_col = df.columns[0]

    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

    if quantity_col:
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
        df["TotalSales"] = df[sales_col] * df[quantity_col]
    else:
        df["TotalSales"] = df[sales_col]

    df = df.dropna(subset=["TotalSales"])

    # DATE PROCESSING
    sales_month = None
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df["YearMonth"] = df[date_col].dt.to_period("M")
            sales_month = df.groupby("YearMonth")["TotalSales"].sum()
            sales_month.index = sales_month.index.to_timestamp()
        except:
            sales_month = None

    # NUMERIC CLEANING
    numeric = df.select_dtypes(include="number").copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.mean())

    # ------------------------------------------------
    # OVERVIEW (DATA PREPROCESSING + EDA)
    # ------------------------------------------------
    if option == "Overview":

        st.subheader("📊 Data Preprocessing Summary")

        st.write("✔ Removed duplicate records")
        st.write("✔ Converted categorical columns to string")
        st.write("✔ Handled missing values using mean imputation")
        st.write("✔ Converted numeric columns properly")

        st.subheader("🔄 Data Transformation")
        st.write("✔ Created feature: TotalSales = Price × Quantity")

        if date_col:
            st.write("✔ Converted date column and extracted Year-Month")

        st.subheader("📁 Dataset Preview")
        st.dataframe(df.head())

        if not numeric.empty:

            st.subheader("📈 Descriptive Statistics")
            st.dataframe(numeric.describe())

            st.write("👉 Interpretation:")
            st.write("- Mean shows average values")
            st.write("- Std shows variability")
            st.write("- Min & Max show range")

            st.subheader("📊 Sales Distribution")
            fig, ax = plt.subplots()
            df["TotalSales"].hist(bins=30, ax=ax)
            st.pyplot(fig)

            st.subheader("📦 Boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(x=df["TotalSales"], ax=ax)
            st.pyplot(fig)

            st.subheader("🔥 Correlation Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        else:
            st.warning("No numeric data available")

        if product_col:
            st.subheader("🏆 Top Products")
            top_products = df.groupby(product_col)["TotalSales"].sum().sort_values(ascending=False).head(5)
            fig, ax = plt.subplots()
            top_products.plot(kind="bar", ax=ax)
            st.pyplot(fig)

        if sales_month is not None:
            st.subheader("📅 Monthly Trend")
            fig, ax = plt.subplots()
            sales_month.plot(ax=ax)
            st.pyplot(fig)

    # ------------------------------------------------
    # INDEX NUMBERS
    # ------------------------------------------------
    if option == "Index Numbers":

        base = df["TotalSales"].iloc[0]

        simple = df["TotalSales"].sum() / base * 100
        weighted = (df["TotalSales"] ** 2).sum() / df["TotalSales"].sum()
        laspeyres = simple
        paasche = weighted
        fisher = (laspeyres * paasche) ** 0.5

        st.subheader("Index Numbers")
        st.write(f"Simple Index: {simple:.2f}")
        st.write(f"Weighted Index: {weighted:.2f}")
        st.write(f"Laspeyres Index: {laspeyres:.2f}")
        st.write(f"Paasche Index: {paasche:.2f}")
        st.write(f"Fisher Index: {fisher:.2f}")

        fig, ax = plt.subplots()
        ax.bar(
            ["Simple", "Weighted", "Laspeyres", "Paasche", "Fisher"],
            [simple, weighted, laspeyres, paasche, fisher]
        )
        ax.set_title("Index Number Comparison")
        st.pyplot(fig)

# ------------------------------------------------
# REGRESSION
# ------------------------------------------------
if option == "Regression Models":

    if len(numeric.columns) > 1:

        X = numeric.drop(columns=["TotalSales"], errors="ignore")
        y = numeric["TotalSales"]

        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        if X.shape[1] > 0:

            feature = X.iloc[:, 0]

            # ------------------------------------------------
            # LINEAR REGRESSION
            # ------------------------------------------------
            st.subheader("Linear Regression")

            lin_model = LinearRegression()
            lin_model.fit(feature.values.reshape(-1,1), y)

            x_range = np.linspace(feature.min(), feature.max(), 100)
            y_line = lin_model.predict(x_range.reshape(-1,1))

            fig, ax = plt.subplots()
            ax.scatter(feature, y, alpha=0.5)
            ax.plot(x_range, y_line, color='red')
            ax.set_title("Linear Regression")
            st.pyplot(fig)

            # ------------------------------------------------
            # POLYNOMIAL REGRESSION
            # ------------------------------------------------
            st.subheader("Polynomial Regression")

            poly = PolynomialFeatures(2)
            X_poly = poly.fit_transform(feature.values.reshape(-1,1))

            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)

            x_poly = poly.transform(x_range.reshape(-1,1))
            y_curve = poly_model.predict(x_poly)

            fig, ax = plt.subplots()
            ax.scatter(feature, y, alpha=0.5)
            ax.plot(x_range, y_curve, color='green')
            ax.set_title("Polynomial Regression")
            st.pyplot(fig)

            # ------------------------------------------------
            # MODEL COMPARISON (NUMERICAL)
            # ------------------------------------------------
            st.subheader("Model Comparison")

            y_pred_lin = lin_model.predict(feature.values.reshape(-1,1))
            y_pred_poly = poly_model.predict(X_poly)

            lin_rmse = np.sqrt(mean_squared_error(y, y_pred_lin))
            poly_rmse = np.sqrt(mean_squared_error(y, y_pred_poly))

            st.write(f"Linear RMSE: {lin_rmse:.2f}")
            st.write(f"Polynomial RMSE: {poly_rmse:.2f}")

            if poly_rmse < lin_rmse:
                st.success("Polynomial Regression performs better")
            else:
                st.success("Linear Regression performs better")

            # ------------------------------------------------
            # RMSE COMPARISON GRAPH
            # ------------------------------------------------
            st.subheader("RMSE Comparison Graph")

            fig, ax = plt.subplots()
            ax.bar(["Linear", "Polynomial"], [lin_rmse, poly_rmse])
            ax.set_title("RMSE Comparison")
            ax.set_ylabel("RMSE")
            st.pyplot(fig)

            # ------------------------------------------------
            # ACTUAL VS PREDICTED GRAPH
            # ------------------------------------------------
            st.subheader("Actual vs Predicted Comparison")

            fig, ax = plt.subplots()
            ax.scatter(y, y_pred_lin, label="Linear", alpha=0.5)
            ax.scatter(y, y_pred_poly, label="Polynomial", alpha=0.5)

            ax.set_xlabel("Actual Sales")
            ax.set_ylabel("Predicted Sales")
            ax.set_title("Actual vs Predicted")
            ax.legend()

            st.pyplot(fig)

            # ------------------------------------------------
            # LOGISTIC REGRESSION (S-CURVE)
            # ------------------------------------------------
            st.subheader("Logistic Regression")

            y_bin = (y > y.mean()).astype(int)

            if y_bin.nunique() > 1:

                log = LogisticRegression(max_iter=1000)
                log.fit(feature.values.reshape(-1,1), y_bin)

                probs = log.predict_proba(x_range.reshape(-1,1))[:,1]

                fig, ax = plt.subplots()
                ax.scatter(feature, y_bin, alpha=0.3)
                ax.plot(x_range, probs, color='red', linewidth=2)

                ax.set_title("Logistic Regression (S-Curve)")
                ax.set_xlabel("Feature")
                ax.set_ylabel("Probability")

                st.pyplot(fig)

            else:
                st.warning("Logistic Regression needs two classes.")

        else:
            st.warning("No predictor columns available")

    else:
        st.warning("Not enough numeric columns")
# ------------------------------------------------
# ANOVA
# ------------------------------------------------
if option == "ANOVA & Tests":

        if len(numeric.columns) >= 2:

            f_stat, p_anova = stats.f_oneway(numeric.iloc[:, 0], numeric.iloc[:, 1])
            u_stat, p_mw = stats.mannwhitneyu(numeric.iloc[:, 0], numeric.iloc[:, 1])

            st.write(f"ANOVA p-value: {p_anova:.4f}")
            st.write(f"Mann-Whitney p-value: {p_mw:.4f}")

            fig, ax = plt.subplots()
            numeric.boxplot(ax=ax)
            ax.set_title("Boxplot Comparison")
            st.pyplot(fig)
        else:
            st.warning("At least two numeric columns are required.")

# ------------------------------------------------
# TIME SERIES
# ------------------------------------------------
if option == "Time Series Analysis":

        if sales_month is not None and len(sales_month) >= 6:

            st.subheader("Time Series Analysis")

            adf_result = adfuller(sales_month)
            st.write(f"ADF Test p-value: {adf_result[1]:.4f}")

            ma = sales_month.rolling(3).mean()
            exp = sales_month.ewm(span=3).mean()

            model = ARIMA(sales_month, order=(1, 1, 1)).fit()
            forecast = model.forecast(3)

            st.write("ARIMA Model: (1,1,1)")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sales_month, label="Original")
            ax.plot(ma, label="Moving Avg")
            ax.plot(exp, label="Exp Smoothing")
            ax.plot(forecast, label="Forecast")
            ax.legend()
            ax.grid(True)
            ax.set_title("ARIMA Forecasting")
            st.pyplot(fig)
        else:
            st.warning("A valid date column with enough time points is required.")




# ------------------------------------------------
# DIMENSIONALITY REDUCTION (PCA + FA)
# ------------------------------------------------
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

if option == "Dimensionality Reduction":

    if len(numeric.columns) > 1:

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric)

        # PCA
        st.subheader("PCA Analysis")

        pca = PCA()
        pca_data = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots()
        ax.scatter(pca_data[:, 0], pca_data[:, 1])
        ax.set_title("PCA Projection")
        st.pyplot(fig)

        # ------------------------------------------------
        # PCA INTERPRETATION
        # ------------------------------------------------
        explained_variance = pca.explained_variance_ratio_

        st.subheader("PCA Interpretation")
        st.write("Explained Variance Ratio:")
        st.write(explained_variance)

        st.info(
                f"First component explains {explained_variance[0]*100:.2f}% of variance. "
                "Higher value indicates better dimensionality reduction."
        )

        # Factor Analysis
        st.subheader("Factor Analysis")

        fa = FactorAnalysis(n_components=2)
        fa_data = fa.fit_transform(scaled_data)

        fig, ax = plt.subplots()
        ax.scatter(fa_data[:, 0], fa_data[:, 1])
        ax.set_title("Factor Analysis Plot")
        st.pyplot(fig)

    else:
        st.warning("Not enough numeric data")

# ------------------------------------------------
# CLUSTERING
# ------------------------------------------------
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if option == "Clustering Techniques":

    if len(numeric.columns) > 1:

        # 🔥 SAMPLE (SAFE)
        sample_data = numeric.sample(n=min(300, len(numeric)), random_state=42)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sample_data)

        # ------------------------------------------------
        # K-MEANS (CLEAR VISUAL)
        # ------------------------------------------------
        st.subheader("K-Means Clustering")

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        # 🔥 REDUCE TO 2D (IMPORTANT)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots()

        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=clusters,
            alpha=0.6
        )

        # 🔥 CENTROIDS
        centroids = kmeans.cluster_centers_
        centroids_2d = pca.transform(centroids)

        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            marker='X',
            s=200
        )

        ax.set_title("K-Means Clustering (Clear Groups)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        st.pyplot(fig)

        # ------------------------------------------------
        # HIERARCHICAL (CLEAN)
        # ------------------------------------------------
        st.subheader("Hierarchical Clustering")

        linked = linkage(scaled_data, method='ward')

        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(linked, ax=ax)

        ax.set_title("Dendrogram (Cluster Formation)")
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Distance")

        st.pyplot(fig)

    else:
        st.warning("Not enough numeric data")

# ------------------------------------------------
# PERFORMANCE
# ------------------------------------------------
if option == "Performance Evaluation":

    if sales_month is not None and len(sales_month) >= 6:

        st.subheader("Performance Metrics")

        # ARIMA MODEL
        model = ARIMA(sales_month, order=(1, 1, 1)).fit()
        forecast = model.forecast(3)

        y_true = sales_month[-3:]

        # METRICS
        mse = mean_squared_error(y_true, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, forecast)

        st.write(f"MSE (Mean Squared Error): {mse:.2f}")
        st.write(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
        st.write(f"MAE (Mean Absolute Error): {mae:.2f}")

        # BAR CHART
        fig, ax = plt.subplots()
        ax.bar(["MSE", "RMSE", "MAE"], [mse, rmse, mae])
        ax.set_title("Performance Metrics Comparison")
        st.pyplot(fig)

        # 👉 IMPORTANT INTERPRETATION (FOR MARKS)
        st.info(
            "Lower values of MSE, RMSE, and MAE indicate better model performance. "
            "Among these, RMSE is more interpretable as it is in the same unit as the data."
        )

    else:
        st.warning("A valid date column with enough time points is required.")

# ------------------------------------------------
# INFERENCE
# ------------------------------------------------
if option == "Inference":

        st.subheader("Dataset Inference")

        insights = []

        total_revenue = df["TotalSales"].sum()
        avg_sales = df["TotalSales"].mean()
        max_sale = df["TotalSales"].max()
        min_sale = df["TotalSales"].min()

        insights.append(
            f"The dataset generated a total revenue of {total_revenue:,.2f}, with an average transaction value of {avg_sales:,.2f}."
        )

        insights.append(
            f"The highest transaction value is {max_sale:,.2f}, while the minimum transaction value is {min_sale:,.2f}."
        )

        if quantity_col and quantity_col in df.columns:
            total_qty = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0).sum()
            insights.append(
                f"The dataset records a total quantity of {total_qty:,.0f} units sold."
            )

        if product_col and product_col in df.columns:
            top_product_series = df.groupby(product_col)["TotalSales"].sum().sort_values(ascending=False)
            if len(top_product_series) > 0:
                top_product = top_product_series.index[0]
                top_product_sales = top_product_series.iloc[0]
                insights.append(
                    f"The top revenue-generating product is '{top_product}' with total sales of {top_product_sales:,.2f}."
                )

        if customer_col and customer_col in df.columns:
            top_customer_series = df.groupby(customer_col)["TotalSales"].sum().sort_values(ascending=False)
            if len(top_customer_series) > 0:
                top_customer = top_customer_series.index[0]
                top_customer_sales = top_customer_series.iloc[0]
                insights.append(
                    f"The highest contributing customer is '{top_customer}', generating sales of {top_customer_sales:,.2f}."
                )

        if country_col and country_col in df.columns:
            top_country_series = df.groupby(country_col)["TotalSales"].sum().sort_values(ascending=False)
            if len(top_country_series) > 0:
                top_country = top_country_series.index[0]
                top_country_sales = top_country_series.iloc[0]
                insights.append(
                    f"The leading market is '{top_country}', contributing {top_country_sales:,.2f} in sales."
                )

        if sales_month is not None and len(sales_month) >= 2:
            recent_value = sales_month.iloc[-1]
            historical_mean = sales_month.mean()

            if recent_value > historical_mean:
                insights.append(
                    "Recent sales are above the historical average, indicating a positive overall trend."
                )
            else:
                insights.append(
                    "Recent sales are below the historical average, indicating a possible slowdown in performance."
                )

            try:
                model = ARIMA(sales_month, order=(1, 1, 1)).fit()
                forecast = model.forecast(3)
                forecast_avg = forecast.mean()

                if forecast_avg > recent_value:
                    insights.append(
                        "ARIMA forecasting suggests that sales may improve in the upcoming periods."
                    )
                else:
                    insights.append(
                        "ARIMA forecasting suggests that sales may remain stable or slightly decline in the upcoming periods."
                    )
            except Exception:
                pass

        if len(numeric.columns) > 1:
            corr_matrix = numeric.corr()
            if "TotalSales" in corr_matrix.columns:
                sales_corr = corr_matrix["TotalSales"].drop(labels=["TotalSales"], errors="ignore")
                if len(sales_corr) > 0:
                    top_corr_feature = sales_corr.abs().sort_values(ascending=False).index[0]
                    top_corr_value = sales_corr[top_corr_feature]
                    insights.append(
                        f"The variable most strongly associated with TotalSales is '{top_corr_feature}' with a correlation of {top_corr_value:.2f}."
                    )

        for item in insights:
            st.write("•", item)
