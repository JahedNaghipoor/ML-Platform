import streamlit as st
import numpy as np
from sklearn import datasets


st.write("""
         # Explore different classifiers
         """)


dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Iris", "Diabetes", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Random Forest", "SVM", "K-Nearest-Neighbour", "Logistic Regression"))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breaset Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Diabetes":
        data = datasets.load_diabetes()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
st.write("Shape of Dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "K-Nearest-Neighbour":
        K = st.sidebar.slider("K", 1, 12)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Max depth", 2, 15)
        n_estimators = st.sidebar.slider("Number of estimators", 1, 100)
        params["Max depth"] = max_depth
        params["Number of estimators"] = n_estimators
    return params


add_parameter_ui(classifier_name)
