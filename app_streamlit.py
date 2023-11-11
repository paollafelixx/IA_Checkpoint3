import streamlit as st
import pandas as pd
import joblib

# Carregue o modelo treinado
model_path = '/content/drive/  sample_data.pkl'
model = joblib.load(model_path)


def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    prediction = model.predict(input_data)
    return prediction[0]


def main():
    st.title('Predição com Modelo Treinado')

    sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
    sepal_width = st.slider('Sepal Width', 2.0, 4.0, 3.0)
    petal_length = st.slider('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.slider('Petal Width', 0.1, 2.5, 1.0)

    if st.button('Fazer Previsão'):
        prediction = predict(sepal_length, sepal_width, petal_length, petal_width)
        st.success(f'A classe prevista é: {prediction}')


if __name__ == '__main__':
    main()
