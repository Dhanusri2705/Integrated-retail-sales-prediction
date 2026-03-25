# =====================================================
# 1 IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# 2 LOAD DATASET
# =====================================================

df = pd.read_excel("Online Retail.xlsx")

print("Dataset Shape:", df.shape)
print(df.head())


# =====================================================
# 3 DATA CLEANING
# =====================================================

df = df.dropna()

df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

df = df[df['Quantity'] > 0]

df = df[df['UnitPrice'] > 0]

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['Month'] = df['InvoiceDate'].dt.month
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')


# =====================================================
# 4 DESCRIPTIVE STATISTICS
# =====================================================

desc_table = pd.DataFrame({
"Metric":[
"Mean Quantity","Mean Unit Price","Mean Total Price",
"Variance Quantity","Variance Unit Price","Variance Total Price"
],

"Value":[
df['Quantity'].mean(),
df['UnitPrice'].mean(),
df['TotalPrice'].mean(),
df['Quantity'].var(),
df['UnitPrice'].var(),
df['TotalPrice'].var()
]
})

print("\nDESCRIPTIVE STATISTICS")
print(desc_table)


# =====================================================
# 5 GROUP-WISE STATISTICS
# =====================================================

group_stats = df.groupby('Country')['TotalPrice'].mean().head()

print("\nAVERAGE SALES BY COUNTRY")
print(group_stats.to_frame(name="Average Sales"))


# =====================================================
# 6 COVARIANCE & CORRELATION
# =====================================================

cov_matrix = df[['Quantity','UnitPrice','TotalPrice']].cov()
corr_matrix = df[['Quantity','UnitPrice','TotalPrice']].corr()

print("\nCOVARIANCE MATRIX")
print(cov_matrix)

print("\nCORRELATION MATRIX")
print(corr_matrix)


# =====================================================
# 7 INDEX NUMBERS
# =====================================================

base_price = df['UnitPrice'].iloc[0]
base_quantity = df['Quantity'].iloc[0]

simple_index = df['UnitPrice'].sum()/base_price*100

weighted_index = (df['UnitPrice']*df['Quantity']).sum()/(base_price*base_quantity)*100

laspeyres = (df['UnitPrice']*base_quantity).sum()/(base_price*base_quantity)*100

paasche = (df['UnitPrice']*df['Quantity']).sum()/(base_price*df['Quantity']).sum()*100

marshall_edge = (
(df['UnitPrice']*(df['Quantity']+base_quantity)).sum()
/
(base_price*(df['Quantity']+base_quantity)).sum()
)*100

fisher = np.sqrt(laspeyres*paasche)

index_table = pd.DataFrame({
"Index Type":[
"Simple Aggregate Index",
"Weighted Aggregate Index",
"Laspeyres Index",
"Paasche Index",
"Marshall Edgeworth Index",
"Fisher Ideal Index"
],

"Value":[
simple_index,
weighted_index,
laspeyres,
paasche,
marshall_edge,
fisher
]
})

print("\nINDEX NUMBERS")
print(index_table)


# =====================================================
# 8 REGRESSION MODEL
# =====================================================

X = df[['Quantity','UnitPrice']]
y = df['TotalPrice']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)


# =====================================================
# 9 POLYNOMIAL REGRESSION
# =====================================================

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(df[['Quantity']])

poly_model = LinearRegression()
poly_model.fit(X_poly,df['TotalPrice'])


# =====================================================
# 10 LOGISTIC REGRESSION
# =====================================================

df['HighValue'] = (df['TotalPrice'] > df['TotalPrice'].median()).astype(int)

X_log = df[['Quantity','UnitPrice']]
y_log = df['HighValue']

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_log,y_log)


# =====================================================
# 11 HYPOTHESIS TESTING
# =====================================================

countries = df['Country'].unique()[:3]

g1 = df[df['Country']==countries[0]]['TotalPrice']
g2 = df[df['Country']==countries[1]]['TotalPrice']
g3 = df[df['Country']==countries[2]]['TotalPrice']

anova = f_oneway(g1,g2,g3)

u_stat,p_val = mannwhitneyu(g1,g2)

test_table = pd.DataFrame({
"Test":["One-Way ANOVA","Mann Whitney U"],
"Statistic":[anova.statistic,u_stat],
"p-value":[anova.pvalue,p_val]
})

print("\nHYPOTHESIS TEST RESULTS")
print(test_table)


# =====================================================
# 12 TIME SERIES PREPARATION
# =====================================================

sales_month = df.groupby('YearMonth')['TotalPrice'].sum()

sales_month.index = sales_month.index.to_timestamp()


# =====================================================
# 13 MOVING AVERAGE
# =====================================================

moving_avg = sales_month.rolling(window=3).mean()

print("\nMOVING AVERAGE")
print(moving_avg)


# =====================================================
# 14 WEIGHTED MOVING AVERAGE
# =====================================================

weights = np.array([0.1,0.3,0.6])

wma = sales_month.rolling(3).apply(lambda x: np.sum(weights*x), raw=True)

print("\nWEIGHTED MOVING AVERAGE")
print(wma)


# =====================================================
# 15 EXPONENTIAL SMOOTHING
# =====================================================

exp_smooth = sales_month.ewm(alpha=0.3).mean()

print("\nEXPONENTIAL SMOOTHING")
print(exp_smooth)


# =====================================================
# 16 ADF TEST
# =====================================================

adf_result = adfuller(sales_month)

print("\nADF TEST RESULT")
print("ADF Statistic:",adf_result[0])
print("p-value:",adf_result[1])


# =====================================================
# 17 ARIMA MODEL
# =====================================================

arima_model = ARIMA(sales_month,order=(1,1,1))

arima_fit = arima_model.fit()

forecast = arima_fit.forecast(steps=3)

forecast_table = forecast.to_frame(name="Predicted Sales")

print("\nARIMA FORECAST")
print(forecast_table)


# =====================================================
# 18 MODEL PERFORMANCE
# =====================================================

mse = mean_squared_error(y_test,pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test,pred)

metrics_table = pd.DataFrame({
"Metric":["MSE","RMSE","MAE"],
"Value":[mse,rmse,mae]
})

print("\nMODEL PERFORMANCE")
print(metrics_table)


# =====================================================
# 19 CUSTOMER SEGMENTATION (K-MEANS)
# =====================================================

customer_data = df.groupby('CustomerID').agg({
'TotalPrice':'sum',
'Quantity':'sum'
})

scaler = StandardScaler()

scaled_data = scaler.fit_transform(customer_data)

kmeans = KMeans(n_clusters=3, random_state=42)

customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

print("\nCUSTOMER SEGMENTS")
print(customer_data.head())


plt.figure()
plt.scatter(customer_data['Quantity'], customer_data['TotalPrice'],
c=customer_data['Cluster'], cmap='viridis')

plt.xlabel("Total Quantity Purchased")
plt.ylabel("Total Spending")
plt.title("Customer Segmentation")
plt.show()


# =====================================================
# 20 TOP SELLING PRODUCTS
# =====================================================

top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

print("\nTOP 10 PRODUCTS")
print(top_products)

plt.figure()
top_products.plot(kind='bar')

plt.title("Top Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")

plt.show()


# =====================================================
# 21 VISUAL ANALYTICS
# =====================================================

plt.figure()
plt.hist(df['TotalPrice'], bins=50)
plt.title("Distribution of Total Sales")
plt.xlabel("Total Price")
plt.ylabel("Frequency")
plt.show()


plt.figure()
sns.heatmap(df[['Quantity','UnitPrice','TotalPrice']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure()
country_sales.plot(kind='bar')
plt.title("Top Countries by Sales")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.show()


plt.figure()
sales_month.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()


plt.figure()
plt.plot(sales_month,label="Original Sales")
plt.plot(moving_avg,label="Moving Average")
plt.legend()
plt.title("Sales Trend with Moving Average")
plt.show()


plt.figure()
plt.plot(sales_month,label="Historical Sales")
plt.plot(forecast,label="Forecast",color='red')
plt.legend()
plt.title("ARIMA Forecast")
plt.show()
