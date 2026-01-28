import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


# Load Dataset

data = pd.read_csv("IRIS.csv")

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

le = LabelEncoder()
y = le.fit_transform(y)


# Train Model

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)


# Streamlit UI

st.title("🌸 Iris Flower Classification App 🌸")
st.write("Enter flower measurements to predict the species")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict Species"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    )

    prediction = model.predict(input_data)
    species = le.inverse_transform(prediction)[0]

    st.success(f"🌼 Predicted Species: **{species}**")
