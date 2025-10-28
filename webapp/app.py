# app.py
import streamlit as st
import pandas as pd

st.title("🧬 BRCATranstypia Demo")
st.write("A mock interface for TCGA-BRCA transcriptomic subtype prediction.")

# Gene list
genes = ["CLEC3A", "HOXB13", "S100A7", "SERPINA6", "VSTM2A", "CST9", "UGT2B11"]

st.subheader("Gene Expression Input")
st.write("Each value corresponds to z-scored expression for the listed genes:")

st.write(", ".join(genes))

# Default sample values
default_values = "-1.278, 0.999, -1.139, -0.013, -0.181, -0.482, 0.214"

# Input field
user_input = st.text_area("Enter comma-separated expression values:", value=default_values)

# Predict button
if st.button("Predict Subtype"):
    st.success("Predicted Subtype: Luminal A  ✅")

# Create small CSV for download
data = {
    "CLEC3A": [-1.278, -0.542],
    "HOXB13": [0.999, 0.654],
    "S100A7": [-1.139, -0.772],
    "SERPINA6": [-0.013, -0.311],
    "VSTM2A": [-0.181, -0.220],
    "CST9": [-0.482, -0.301],
    "UGT2B11": [0.214, 0.140],
}
df = pd.DataFrame(data)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download sample CSV",
    data=csv,
    file_name="sample_test_data.csv",
    mime="text/csv",
)
