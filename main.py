pip install sklearn
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import AgglomerativeClustering
dataFrame=pd.read_csv('Mall_Customers.csv')

st.write("""
# Customer Segmentation App

This web app shows the number of clusters


""")

st.sidebar.header('User Input Features')


dataFrame = dataFrame.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_score'})

Clusters = st.sidebar.slider('k', 1, 10, 5)
Annual_Income = st.sidebar.slider('Annual_Income', 15,137,35)
Spending_score = st.sidebar.slider('Spending_score', 1,99,25)      
def user_input_features():
    #Clusters = st.sidebar.slider('k', 1, 10, 5)
    #Annual_Income = st.sidebar.slider('Annual_Income', 15,137,35)
    #Spending_score = st.sidebar.slider('Spending_score', 1,99,25)      
    data = {'Clusters': Clusters,
            'Annual_Income': Annual_Income,
            'Spending_score': Spending_score
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


kmeans = KMeans(n_clusters = Clusters,random_state = 111)
kmeans.fit(dataFrame.iloc[:,2:5])

cluster_labels = kmeans.fit_predict(dataFrame.iloc[:,2:5])

preds = kmeans.labels_
kmeans_df = pd.DataFrame(dataFrame.iloc[:,2:5])
kmeans_df['KMeans_Clusters'] = preds
kmeans_df.head(5)



#visulization of clusters  income vs spending score
sns.scatterplot(kmeans_df['Annual_Income'],kmeans_df['Spending_score'],hue='KMeans_Clusters',data=kmeans_df,palette="deep")

# plt.scatter(df.Attack, df.Defense, c=df.c, alpha = 0.6, s=10)
plt.title("Annual_Income  vs Spending_score", fontsize=15)
plt.xlabel("Annual_Income", fontsize=12)
plt.ylabel("Spending_score", fontsize=12)
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.button('scatter plot'):
    st.header('Annual income vs spending score')
    
    
    #sns.axes_style("white"):
    
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.scatterplot(kmeans_df['Annual_Income'],kmeans_df['Spending_score'],hue='KMeans_Clusters',data=kmeans_df,palette="deep")
st.pyplot()


# Saving the model
import pickle
pickle.dump(kmeans, open('kmeans.pkl', 'wb'))


