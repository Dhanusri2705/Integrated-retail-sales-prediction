# ===============================
# 1. Import Required Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# 2. Load Dataset
# ===============================

df = pd.read_excel("Online Retail.xlsx")

print("Dataset Shape:", df.shape)
print(df.head())


# ===============================
# 3. Dataset Information
# ===============================

print(df.info())

print("Missing Values:")
print(df.isnull().sum())


# ===============================
# 4. Data Cleaning
# ===============================

# Remove missing values
df = df.dropna()

# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

# Remove negative quantities
df = df[df['Quantity'] > 0]

# Remove negative prices
df = df[df['UnitPrice'] > 0]


# ===============================
# 5. Feature Engineering
# ===============================

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract Month
df['Month'] = df['InvoiceDate'].dt.month


# ===============================
# 6. Exploratory Data Analysis
# ===============================

# Top Countries by Orders
top_countries = df['Country'].value_counts().head(10)

plt.figure()
top_countries.plot(kind='bar')
plt.title("Top 10 Countries by Orders")
plt.xlabel("Country")
plt.ylabel("Number of Orders")
plt.show()


# Top Selling Products
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_products.plot(kind='bar')
plt.title("Top Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.show()


# Monthly Sales Trend
monthly_sales = df.groupby('Month')['TotalPrice'].sum()

plt.figure()
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()


# ===============================
# 7. Sales by Country
# ===============================

country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure()
country_sales.plot(kind='bar')
plt.title("Top Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.show()


# ===============================
# 8. RFM Customer Segmentation
# ===============================

# Create snapshot date
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

print(rfm.head())


# ===============================
# 9. Customer Segmentation
# ===============================

rfm['CustomerSegment'] = pd.qcut(
    rfm['Monetary'],
    q=3,
    labels=['Low Value', 'Medium Value', 'High Value']
)

print(rfm.head())


# ===============================
# 10. Customer Segment Visualization
# ===============================

segment_counts = rfm['CustomerSegment'].value_counts()

plt.figure()
segment_counts.plot(kind='bar')
plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.show()


# ===============================
# 11. Top Customers
# ===============================

top_customers = rfm.sort_values('Monetary', ascending=False).head(10)

print("Top Customers:")
print(top_customers)


# ===============================
# 12. Business Insights
# ===============================

print("Total Transactions:", df['InvoiceNo'].nunique())
print("Total Customers:", df['CustomerID'].nunique())
print("Total Revenue:", df['TotalPrice'].sum())
