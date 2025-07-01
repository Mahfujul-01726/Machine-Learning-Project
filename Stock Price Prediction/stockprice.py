from flask import Flask, render_template, request
import joblib
import yfinance as yf
import pandas as pd

app = Flask(__name__)


model = joblib.load("stock_price_model.pkl")

@app.route('/')
def home():
    return render_template('mlindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ticker = request.form['ticker']  # Get ticker from form input
        if not ticker.strip():
            return render_template('mlindex.html', prediction="Please enter a valid ticker symbol.")

        # Fetch recent data for the ticker
        try:
            data = yf.download(ticker, period="1y")
            if data.empty:
                raise ValueError("No data available for the ticker.")
        except Exception as e:
            return render_template('mlindex.html', prediction=f"Error fetching data: {str(e)}")

        # Prepare the data for prediction
        data['Days'] = range(len(data))
        future_day = len(data) + 1

        # Predict the next day's price
        predicted_price = model.predict([[future_day]])  # Predict price
        scalar_price = predicted_price[0][0] if predicted_price.ndim > 1 else predicted_price[0]  

        # Format the predicted price for display
        formatted_price = f"${scalar_price:.2f}"

        return render_template('result.html', prediction=formatted_price, ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True)