import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


# ------------------------------
# GLOBAL DATA
# ------------------------------

df = None


# ------------------------------
# LOAD DATASET
# ------------------------------

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

    update_kpis()

    messagebox.showinfo("Success", "Dataset Loaded Successfully")


# ------------------------------
# KPI CARDS
# ------------------------------

def update_kpis():

    if df is None:
        return

    revenue = df['TotalPrice'].sum()
    orders = df['InvoiceNo'].nunique()
    customers = df['CustomerID'].nunique()
    avg_order = df['TotalPrice'].mean()

    revenue_var.set(f"₹ {revenue:,.0f}")
    orders_var.set(orders)
    customers_var.set(customers)
    avg_order_var.set(f"{avg_order:.2f}")


# ------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------

def descriptive_statistics():

    if df is None:
        messagebox.showwarning("Warning","Load dataset first")
        return

    stats = df[['Quantity','UnitPrice','TotalPrice']].describe()

    text.delete(1.0, END)

    text.insert(END, "DESCRIPTIVE STATISTICS\n\n")
    text.insert(END, stats.to_string())


# ------------------------------
# SALES TREND
# ------------------------------

def sales_trend():

    if df is None:
        return

    sales_month = df.groupby('YearMonth')['TotalPrice'].sum()
    sales_month.index = sales_month.index.to_timestamp()

    fig.clear()

    ax = fig.add_subplot(111)

    sales_month.plot(ax=ax)

    ax.set_title("Monthly Sales Trend")

    canvas.draw()


# ------------------------------
# TOP PRODUCTS
# ------------------------------

def top_products():

    if df is None:
        return

    products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

    fig.clear()

    ax = fig.add_subplot(111)

    products.plot(kind='bar', ax=ax)

    ax.set_title("Top Selling Products")

    canvas.draw()


# ------------------------------
# CORRELATION
# ------------------------------

def correlation():

    if df is None:
        return

    fig.clear()

    ax = fig.add_subplot(111)

    corr = df[['Quantity','UnitPrice','TotalPrice']].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    ax.set_title("Correlation Heatmap")

    canvas.draw()


# ------------------------------
# CUSTOMER SEGMENTATION
# ------------------------------

def segmentation():

    if df is None:
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

    ax.scatter(customer_data['Quantity'],
               customer_data['TotalPrice'],
               c=customer_data['Cluster'],
               cmap='viridis')

    ax.set_title("Customer Segmentation")

    canvas.draw()


# ------------------------------
# FORECAST
# ------------------------------

def forecast():

    if df is None:
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


# ------------------------------
# INFERENCE
# ------------------------------

def inference():

    if df is None:
        return

    text.delete(1.0, END)

    top_country = df.groupby('Country')['TotalPrice'].sum().idxmax()
    top_product = df.groupby('Description')['Quantity'].sum().idxmax()
    avg_sales = df['TotalPrice'].mean()

    text.insert(END, "DATASET INSIGHTS\n\n")

    text.insert(END, f"Average Order Value: {avg_sales:.2f}\n\n")
    text.insert(END, f"Top Revenue Country: {top_country}\n\n")
    text.insert(END, f"Most Popular Product: {top_product}\n\n")

    text.insert(END, "Business Insight:\n")
    text.insert(END, "The dataset indicates strong purchasing behavior in the leading country. ")
    text.insert(END, "The identified product shows highest demand and can be prioritized for marketing.")


# ------------------------------
# MAIN WINDOW
# ------------------------------

root = Tk()

root.title("Retail Predictive Analytics Dashboard")

root.geometry("1200x700")


# HEADER

header = Label(root,
               text="Retail Predictive Analytics Dashboard",
               font=("Arial",20,"bold"))

header.pack(pady=10)


# KPI FRAME

kpi_frame = Frame(root)
kpi_frame.pack()

revenue_var = StringVar()
orders_var = StringVar()
customers_var = StringVar()
avg_order_var = StringVar()

Label(kpi_frame, text="Total Revenue").grid(row=0,column=0,padx=30)
Label(kpi_frame, textvariable=revenue_var,font=("Arial",12,"bold")).grid(row=1,column=0)

Label(kpi_frame, text="Orders").grid(row=0,column=1,padx=30)
Label(kpi_frame, textvariable=orders_var,font=("Arial",12,"bold")).grid(row=1,column=1)

Label(kpi_frame, text="Customers").grid(row=0,column=2,padx=30)
Label(kpi_frame, textvariable=customers_var,font=("Arial",12,"bold")).grid(row=1,column=2)

Label(kpi_frame, text="Avg Order Value").grid(row=0,column=3,padx=30)
Label(kpi_frame, textvariable=avg_order_var,font=("Arial",12,"bold")).grid(row=1,column=3)


# BUTTONS

button_frame = Frame(root)
button_frame.pack(pady=10)

Button(button_frame,text="Load Dataset",command=load_dataset).grid(row=0,column=0,padx=5)
Button(button_frame,text="Descriptive Statistics",command=descriptive_statistics).grid(row=0,column=1,padx=5)
Button(button_frame,text="Sales Trend",command=sales_trend).grid(row=0,column=2,padx=5)
Button(button_frame,text="Top Products",command=top_products).grid(row=0,column=3,padx=5)
Button(button_frame,text="Correlation",command=correlation).grid(row=0,column=4,padx=5)
Button(button_frame,text="Customer Segmentation",command=segmentation).grid(row=0,column=5,padx=5)
Button(button_frame,text="Forecast",command=forecast).grid(row=0,column=6,padx=5)
Button(button_frame,text="Inference",command=inference).grid(row=0,column=7,padx=5)


# GRAPH AREA

fig = plt.Figure(figsize=(7,5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=LEFT)


# TEXT AREA

text = Text(root,width=40)
text.pack(side=RIGHT,padx=10)


root.mainloop()
