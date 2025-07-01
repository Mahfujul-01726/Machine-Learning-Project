from flask import Flask, render_template, request
from fastai.vision.all import *
import os
import pathlib
import torch
from fastai.learner import load_learner

# Fix PosixPath issue on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load trained model
model_path = 'tomato_disease_classifier.pkl'  # Ensure model.pkl is in the same directory
learn = load_learner(model_path, cpu=True)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Load and predict
    img = PILImage.create(file_path)
    pred, _, probs = learn.predict(img)
    
    return render_template('index.html', prediction=pred, confidence=probs.max().item(), image=file_path)

if __name__ == '__main__':
    app.run(debug=True)
