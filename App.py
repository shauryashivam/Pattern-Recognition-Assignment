import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

#import pickle
#import shap

# -----------------
# Importing the data sets
from tf.keras.datasets import mnist
(x_train1,y_train1),(x_test1,y_test1)=mnist.load_data()
x_train1=x_train1.reshape(60000,784)
x_test1=x_test1.reshape(10000,784)

##### EXTRACTING 0 and 1 from the DATASET
x_train=[]
y_train=[]

for i in range(60000):
  if y_train1[i]==0 or y_train1[i]==1:
    temp=[]
    for j in range(784):
      temp.append(x_train1[i][j])
    x_train.append(temp)
    y_train.append(y_train1[i])

x_test=[]
y_test=[]

for i in range(10000):
  if y_test1[i]==0 or y_test1[i]==1:
    temp=[]
    for j in range(784):
      temp.append(x_test1[i][j])
    x_test.append(temp)
    y_test.append(y_test1[i])

x_train=np.matrix(x_train)
y_train=np.array(y_train)
x_test=np.matrix(x_test)
y_test=np.array(y_test)


nb_model=GaussianNB()

fit_nb=nb_model.fit(x_train,y_train)

# ------------------

# -------------------
# Saving the model

#saved_model = pickle.dumps(model)
# -------------------
# -------------------

# Setting of Background
page_bg_img = '''
<style>
body {
background-image: url("https://img.freepik.com/free-vector/bright-background-with-dots_1055-3132.jpg?size=338&ext=jpg&ga=GA1.2.1846610357.1604275200");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_option('deprecation.showPyplotGlobalUse', False)

# Writing on Application
st.write("""
# Pattern Recognition Assignment App
This app predicts the **Authenticity of Bank Notes**!
""")
st.sidebar.title("Prediction App")

nav = st.sidebar.radio("", ["Home", "Answer 1", "Answer 2", "Prediction"])
# ---------------
# ---------------
# Home Page
if nav == "Home":
    st.write("## Description of Predictor App")
    st.write("### The prices of the house indicated by the variable MEDV is our target variable and the remaining are the feature variables based on which we will predict the value of a house.")
    st.write("### The App predicts the price of house after giving different values as input to different features")

    # st.write("## Dataset Description")

    st.write(mnist.DESCR)

  #  if st.checkbox("Show Tabulated"):
   #     st.table()

# ---------------

# ---------------
# Visualization of Data
if nav == "Data Visualisation":
    st.sidebar.write("# Choose From the following")
    st.header("Visualisation")
    st.write("### Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data ")

    if st.sidebar.checkbox("Dist Plot"):
        st.write("## Dist Plot")
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.distplot(boston['MEDV'], bins=30)
        st.pyplot()

    if st.sidebar.checkbox("Correlation Matrix"):
        st.write("## Correlation Matrix")
        correlation_matrix = boston.corr().round(2)
        sns.heatmap(data=correlation_matrix, annot=True)
        st.pyplot()

    if st.sidebar.checkbox("Histogram"):
        st.write("## Histogram")
        boston.hist(edgecolor='black', figsize=(18, 12))
        st.pyplot()


# ------------

# ------------
# Predictoin Application
if nav == "Prediction":
    st.write("## Prediction of  Median value of owner-occupied homes in $1000's")
    st.sidebar.header("Specify Input Parameters")

    val1 = st.sidebar.slider("Per capita crime rate by town", float(
        boston.CRIM.min()), float(boston.CRIM.max()), float(boston.CRIM.mean()))
    val2 = st.sidebar.slider("Nitric oxides concentration (parts per 10 million)", float(
        boston.NOX.min()), float(boston.NOX.max()), float(boston.NOX.mean()))
    val3 = st.sidebar.slider("Average number of rooms per dwelling", float(
        boston.RM.min()), float(boston.RM.max()), float(boston.RM.mean()))
    val4 = st.sidebar.slider("Proportion of owner-occupied units built prior to 1940",
                             float(boston.AGE.min()), float(boston.AGE.max()), float(boston.AGE.mean()))
    
    val = [[val1, val2, val3, val4]]
    model_from_pickle = saved_model
    prediction = model_from_pickle.predict(val)

    if st.button("Predict"):
        st.success(f"Rate is {(prediction)}")

# Explaining the model's predictions using SHAP values
    #explainer = shap.TreeExplainer(model)
   # shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    #shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')

    plt.title('Feature importance based on SHAP values (Bar)')
    #shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
