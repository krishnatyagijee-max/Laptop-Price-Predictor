
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

model = joblib.load("laptop_price_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("💻 Laptop Price Predictor")

companies = ["Acer","Apple","Asus","Chuwi","Dell","Fujitsu","Google","HP","Huawei",
             "LG","Lenovo","MSI","Mediacom","Microsoft","Razer","Samsung","Toshiba","Vero","Xiaomi"]

types = ["Notebook","Gaming","Ultrabook","2 in 1 Convertible","Workstation","Netbook"]

opsys = ["Windows 10","Windows 7","macOS","Mac OS X","Linux","No OS","Chrome OS","Android"]

cpu_brands = ["Intel","AMD","Samsung"]
cpu_families = ["Core i3","Core i5","Core i7","Ryzen","Celeron","Pentium","Xeon","Atom","Other"]
gpu_brands = ["Intel","AMD","Nvidia","ARM"]

company = st.selectbox("Company", companies)
typename = st.selectbox("Type", types)
ops = st.selectbox("Operating System", opsys)

ram = st.selectbox("RAM (GB)", [2,4,6,8,12,16,24,32,64])
weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0)
inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, 15.6)

ssd = st.selectbox("SSD (GB)", [0,128,256,512,1024,2048])
hdd = st.selectbox("HDD (GB)", [0,500,1000,2000])
flash = st.selectbox("Flash Storage (GB)", [0,8,16,32,64,128,256])
hybrid = st.selectbox("Hybrid (GB)", [0,512,1024])

touch = st.selectbox("Touch Screen", ["No","Yes"])
ips = st.selectbox("IPS Panel", ["No","Yes"])

ppi = st.number_input("PPI", 50.0, 400.0, 141.0)

cpu_brand = st.selectbox("CPU Brand", cpu_brands)
cpu_family = st.selectbox("CPU Family", cpu_families)
clock = st.number_input("CPU Clock Speed (GHz)", 0.5, 6.0, 2.5)

gpu_brand = st.selectbox("GPU Brand", gpu_brands)

if st.button("Predict Price"):
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    def put(col, value):
        if col in row.columns:
            row.at[0, col] = value

    # numeric
    for c,v in {
        "Inches":inches,"Ram":ram,"Weight":weight,"SSD":ssd,"HDD":hdd,
        "Flash":flash,"Hybrid":hybrid,
        "TouchScreen":1 if touch=="Yes" else 0,
        "IPS":1 if ips=="Yes" else 0,
        "PPI":ppi,"Clock_speed":clock
    }.items():
        put(c,v)

    # one-hot helper
    for prefix,val in [
        ("Company",company),
        ("TypeName",typename),
        ("OpSys",ops),
        ("Cpu_brand",cpu_brand),
        ("Cpu_family",cpu_family),
        ("Gpu_brand",gpu_brand)
    ]:
        col=f"{prefix}_{val}"
        put(col,1)

    x = scaler.transform(row)
    pred = model.predict(x)[0]

    st.success(f"Estimated Laptop Price: €{pred:,.2f}")
