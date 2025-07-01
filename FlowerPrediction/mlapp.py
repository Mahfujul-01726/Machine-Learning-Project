from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (model is in the same directory)
model = joblib.load("irismodel.pkl")

@app.route('/')
def home():
    return render_template('mlindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare input features for the model
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_features)
        species = ['setosa', 'versicolor', 'virginica']

        # Render result
        return render_template('result.html', prediction=species[prediction[0]])
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
