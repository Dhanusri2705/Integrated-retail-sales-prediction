# =====================================================
# IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# GLOBAL VARIABLES
# =====================================================

df = None


# =====================================================
# FUNCTION: LOAD DATASET
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

    text_box.delete(1.0, END)
    text_box.insert(END, "Dataset Loaded Successfully\n")
    text_box.insert(END, f"Rows: {df.shape[0]}  Columns: {df.shape[1]}\n")


# =====================================================
# FUNCTION: SHOW STATISTICS
# =====================================================

def show_statistics():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    stats = df[['Quantity','UnitPrice','TotalPrice']].describe()

    text_box.delete(1.0, END)
    text_box.insert(END, "DESCRIPTIVE STATISTICS\n\n")
    text_box.insert(END, stats.to_string())


# =====================================================
# FUNCTION: TOP PRODUCTS GRAPH
# =====================================================

def top_products():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

    fig.clear()

    ax = fig.add_subplot(111)

    top_products.plot(kind='bar', ax=ax)

    ax.set_title("Top Selling Products")
    ax.set_xlabel("Product")
    ax.set_ylabel("Quantity")

    canvas.draw()


# =====================================================
# FUNCTION: CORRELATION HEATMAP
# =====================================================

def correlation():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    fig.clear()

    ax = fig.add_subplot(111)

    corr = df[['Quantity','UnitPrice','TotalPrice']].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    ax.set_title("Correlation Heatmap")

    canvas.draw()


# =====================================================
# FUNCTION: CUSTOMER SEGMENTATION
# =====================================================

def segmentation():

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

    fig.clear()

    ax = fig.add_subplot(111)

    scatter = ax.scatter(customer_data['Quantity'],
                         customer_data['TotalPrice'],
                         c=customer_data['Cluster'],
                         cmap='viridis')

    ax.set_title("Customer Segmentation")
    ax.set_xlabel("Quantity Purchased")
    ax.set_ylabel("Total Spending")

    canvas.draw()


# =====================================================
# FUNCTION: SALES TREND
# =====================================================

def sales_trend():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    sales_month = df.groupby('YearMonth')['TotalPrice'].sum()

    sales_month.index = sales_month.index.to_timestamp()

    fig.clear()

    ax = fig.add_subplot(111)

    sales_month.plot(ax=ax)

    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")

    canvas.draw()


# =====================================================
# FUNCTION: FORECAST
# =====================================================

def forecast():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    sales_month = df.groupby('YearMonth')['TotalPrice'].sum()

    sales_month.index = sales_month.index.to_timestamp()

    model = ARIMA(sales_month, order=(1,1,1))

    model_fit = model.fit()

    future = model_fit.forecast(steps=3)

    fig.clear()

    ax = fig.add_subplot(111)

    ax.plot(sales_month, label="Historical")

    ax.plot(future, label="Forecast", color="red")

    ax.legend()

    ax.set_title("Sales Forecast")

    canvas.draw()


# =====================================================
# FUNCTION: INFERENCE
# =====================================================

def inference():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    top_country = df.groupby('Country')['TotalPrice'].sum().idxmax()

    top_product = df.groupby('Description')['Quantity'].sum().idxmax()

    avg_sales = df['TotalPrice'].mean()

    text_box.delete(1.0, END)

    text_box.insert(END, "DATASET INSIGHTS\n\n")

    text_box.insert(END, f"Average Transaction Value: {avg_sales:.2f}\n\n")

    text_box.insert(END, f"Top Revenue Country: {top_country}\n\n")

    text_box.insert(END, f"Most Popular Product: {top_product}\n\n")

    text_box.insert(END, "Business Insight:\n")
    text_box.insert(END, "The dataset shows strong purchasing behavior in the top country. ")
    text_box.insert(END, "The identified top product has the highest demand, indicating ")
    text_box.insert(END, "potential opportunities for targeted marketing and inventory planning.")


# =====================================================
# GUI WINDOW
# =====================================================

root = Tk()

root.title("Retail Predictive Analytics Dashboard")

root.geometry("1000x600")


# LEFT PANEL BUTTONS

frame = Frame(root)

frame.pack(side=LEFT, padx=10)


Button(frame, text="Load Dataset", width=20, command=load_dataset).pack(pady=5)

Button(frame, text="Statistics", width=20, command=show_statistics).pack(pady=5)

Button(frame, text="Top Products", width=20, command=top_products).pack(pady=5)

Button(frame, text="Correlation", width=20, command=correlation).pack(pady=5)

Button(frame, text="Customer Segmentation", width=20, command=segmentation).pack(pady=5)

Button(frame, text="Sales Trend", width=20, command=sales_trend).pack(pady=5)

Button(frame, text="Forecast", width=20, command=forecast).pack(pady=5)

Button(frame, text="Inference", width=20, command=inference).pack(pady=5)


# GRAPH AREA

fig = plt.Figure(figsize=(6,5))

canvas = FigureCanvasTkAgg(fig, master=root)

canvas.get_tk_widget().pack(side=LEFT)


# TEXT OUTPUT AREA

text_box = Text(root, width=40)

text_box.pack(side=RIGHT, padx=10)


root.mainloop()
