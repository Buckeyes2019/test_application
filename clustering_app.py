#%%
import streamlit as st
import pandas as pd
#import plotly.express as px
from sklearn.cluster import KMeans

# %%
st.title("Machine Learning: Clustering Application")
# %%
df = st.file_uploader("Upload your data (.csv files only)", type=["csv"])
var2 = st.selectbox('Select Number of Cluster', [1,2,3,4,5,6,7,8])
var1 = st.slider('Number of Clusters', min_value=2, max_value=8, value=4, step=1)
#var2 = st.slider('Speed vs quality of summary (1 is fastest)', min_value=1, max_value=8, value=4, step=1)
submit = st.button('Execute')
if submit:
    df1 = df.iloc[:, 0:3]
    model = KMeans(n_clusters=var1, random_state=42)
    cluster = model.fit(df1)
    df1["cluster"] = cluster.predict(df1)
    st.dataframe(df1.head)
    #fig = px.scatter_3d(df1, x=first_column, y=second_column, z=third_column, color=df1.cluster)
    #fig.show()

#def Columns_For_Clustering(first_column, second_column, third_column):
    #df1 = df.iloc[:, 0:3]
    #print(df1.columns)
    
#def Run_Clustering(number_of_clusters):
  #  pd.set_option('mode.chained_assignment', None)
   # model = KMeans(n_clusters=var1, random_state=42)
   # cluster = model.fit(df1)
   # df1["cluster"] = cluster.predict(df1)
   # df1.head
    
#def Visualize_Clusters():
 #   fig = px.scatter_3d(df1, x=first_column, y=second_column, z=third_column, color=df1.cluster)
  #  fig.show()




# %%
#filepath_to_data = <<value>>  # filepath_to_data = 'C:/Desktop/FuturesProject/data.csv'
#first_column = <<value>>  # first_column = 'U.S. Interest'
#second_column = <<value>>  # second_column = 'Significance'
#third_column = <<value>>  # third_column = 'Likelihood'
#number_of_clusters = <<value>>  # number_of_clusters = 4
#saved_data_name = <<value>>  

# %%
#def Import_Data(filepath_to_data):
 #   global df
  #  df = pd.read_csv(filepath_to_data)
   # return df

#def Columns_For_Clustering(first_column, second_column, third_column):
 #   global df1
  #  df1 = df[[first_column, second_column, third_column]]
   # print(df1.columns)
    
#def Run_Clustering(number_of_clusters):
 #   pd.set_option('mode.chained_assignment', None)
  #  model = KMeans(n_clusters=number_of_clusters, random_state=42)
   # cluster = model.fit(df1)
    #df1["cluster"] = cluster.predict(df1)
    #df1.head
    
#def Visualize_Clusters():
 #   fig = px.scatter_3d(df1, x=first_column, y=second_column, z=third_column, color=df1.cluster)
  #  fig.show()
    
#def Save_Cluster_Data(saved_data_name):
  #  df1.to_csv(saved_data_name)
