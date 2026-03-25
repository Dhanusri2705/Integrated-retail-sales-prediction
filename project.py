# =====================================================
# IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tkinter import *
from tkinter import filedialog, messagebox

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# GLOBAL DATAFRAME
# =====================================================

df = None


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():
    
    global df
    
    file = filedialog.askopenfilename()
    
    if file == "":
        return
    
    df = pd.read_excel(file)
    
    df.dropna(inplace=True)
    
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    
    messagebox.showinfo("Success", "Dataset Loaded Successfully")


# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

def show_statistics():
    
    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return
    
    stats = df[['Quantity','UnitPrice','TotalPrice']].describe()
    
    print("\nDESCRIPTIVE STATISTICS")
    print(stats)


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def show_correlation():
    
    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return
    
    plt.figure()
    
    corr = df[['Quantity','UnitPrice','TotalPrice']].corr()
    
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    
    plt.title("Correlation Heatmap")
    
    plt.show()


# =====================================================
# TOP PRODUCTS
# =====================================================

def top_products():
    
    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return
    
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    
    print("\nTOP PRODUCTS")
    print(top_products)
    
    plt.figure()
    
    top_products.plot(kind='bar')
    
    plt.title("Top Selling Products")
    
    plt.xlabel("Product")
    plt.ylabel("Quantity Sold")
    
    plt.show()


# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

def customer_segmentation():
    
    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return
    
    customer_data = df.groupby('CustomerID').agg({
        'TotalPrice':'sum',
        'Quantity':'sum'
    })
    
    scaler = StandardScaler()
    
    scaled = scaler.fit_transform(customer_data)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    customer_data['Cluster'] = kmeans.fit_predict(scaled)
    
    plt.figure()
    
    plt.scatter(customer_data['Quantity'],
                customer_data['TotalPrice'],
                c=customer_data['Cluster'],
                cmap='viridis')
    
    plt.xlabel("Quantity Purchased")
    plt.ylabel("Total Spending")
    
    plt.title("Customer Segmentation")
    
    plt.show()


# =====================================================
# MONTHLY SALES TREND
# =====================================================

def monthly_sales():
    
    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return
    
    sales_month = df.groupby('YearMonth')['TotalPrice'].sum()
    
    sales_month.index = sales_month.index.to_timestamp()
    
    plt.figure()
    
    sales_month.plot()
    
    plt.title("Monthly Sales Trend")
    
    plt.xlabel("Month")
    plt.ylabel("Sales")
    
    plt.show()


# =====================================================
# ARIMA FORECAST
# =====================================================

def arima_forecast():
    
    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return
    
    sales_month = df.groupby('YearMonth')['TotalPrice'].sum()
    
    sales_month.index = sales_month.index.to_timestamp()
    
    model = ARIMA(sales_month, order=(1,1,1))
    
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=3)
    
    print("\nFORECAST")
    print(forecast)
    
    plt.figure()
    
    plt.plot(sales_month, label="Historical Sales")
    
    plt.plot(forecast, label="Forecast", color="red")
    
    plt.legend()
    
    plt.title("Sales Forecast")
    
    plt.show()


# =====================================================
# GUI WINDOW
# =====================================================

root = Tk()

root.title("Retail Predictive Analytics Dashboard")

root.geometry("500x500")


title = Label(root,
              text="Retail Predictive Analytics Tool",
              font=("Arial",16,"bold"))

title.pack(pady=20)


btn1 = Button(root,
              text="Load Dataset",
              width=25,
              command=load_dataset)

btn1.pack(pady=5)


btn2 = Button(root,
              text="Show Descriptive Statistics",
              width=25,
              command=show_statistics)

btn2.pack(pady=5)


btn3 = Button(root,
              text="Correlation Heatmap",
              width=25,
              command=show_correlation)

btn3.pack(pady=5)


btn4 = Button(root,
              text="Top Selling Products",
              width=25,
              command=top_products)

btn4.pack(pady=5)


btn5 = Button(root,
              text="Customer Segmentation",
              width=25,
              command=customer_segmentation)

btn5.pack(pady=5)


btn6 = Button(root,
              text="Monthly Sales Trend",
              width=25,
              command=monthly_sales)

btn6.pack(pady=5)


btn7 = Button(root,
              text="ARIMA Forecast",
              width=25,
              command=arima_forecast)

btn7.pack(pady=5)


root.mainloop()
