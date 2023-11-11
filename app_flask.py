from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model_path = '/content/drive/  sample_data.pkl'
model = joblib.load(model_path)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame(data)
        prediction = model.predict(input_data)
        result = prediction[0]

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
