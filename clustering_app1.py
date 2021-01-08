#%%
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %%
st.title("Machine Learning: Clustering Application")
st.subheader("The purpose of this application is to allow you to run the K-means clustering alogrithm on your own data without having to write any code.") 

st.write("_The app is intended for educational use, please do not upload any classified, proprietary, or sensitive information._")

st.write("The first step is to prepare your data. To do so, please put your data into a format like the example given below. You can include as many rows as you like but only three columns.")
demo = {'Category_1': [0.65, 0.75, 0.52, 0.14], 'Category_2': [0.56, 0.23, 0.76, 0.92], 'Category_3': [0.23,0.34, 0.11, 0.64]}
demo1 = pd.DataFrame(demo)
st.dataframe(demo1)

st.write("The next step is to save your data as a comma separated value (csv) file and then drag and drop the file into the space below.")
df = st.file_uploader("Upload your data (.csv files only)", type=["csv"])

st.write("Please select how many clusters you would like to create and click the 'Generate' button.") 

var1 = st.slider('Number of Clusters', min_value=2, max_value=8, value=4, step=1)

submit = st.button('Generate')
if submit:
    st.write("Below you can view the output as a datatable, a bar chart, and 3 dimensional scatterplot. You can also download the new datatable to your computer.")
    df2 = pd.read_csv(df)    
    df1 = df2.iloc[:, 0:3]
    model = KMeans(n_clusters=var1, random_state=42)
    cluster1 = model.fit(df1)
    df1["cluster"] = cluster1.predict(df1)
    st.dataframe(df1)
    
    st.bar_chart(df1['cluster'].value_counts())
    col1 = df1.iloc[:, 0:1]
    col2 = df1.iloc[:, 1:2]
    col3 = df1.iloc[:, 2:3]
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(col1, col2, col3, c=df1.cluster,alpha=1,edgecolors="#000000", s=50)
    ax.set_xlabel('Category 1')
    ax.set_ylabel('Category 2')
    ax.set_zlabel('Category 3')
    st.pyplot(fig)

    csv = df1.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

    st.subheader("Thanks for trying our app!")
