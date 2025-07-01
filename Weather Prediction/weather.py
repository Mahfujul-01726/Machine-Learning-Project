from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# OpenWeather API configuration
API_KEY = "4b1f9e2e23e7a3ebf8287e2b14b4023c"  # Replace with your API key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@app.route('/')
def home():
    return render_template('windex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        city = request.form['city']  # Get city name from user input
        if not city.strip():
            return render_template('windex.html', prediction="Please enter a valid city name.")

        # Fetch weather data from OpenWeather API
        try:
            params = {
                'q': city,
                'appid': API_KEY,
                'units': 'metric'  # Fetch temperature in Celsius
            }
            response = requests.get(BASE_URL, params=params)
            data = response.json()

            if response.status_code != 200:
                return render_template('windex.html', prediction=f"Error: {data.get('message', 'Unknown error')}")

            # Extract required information
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            weather = data['weather'][0]['description']

            # Prepare prediction or output message
            prediction = f"The current temperature in {city.title()} is {temperature}°C with {weather}. Humidity is {humidity}%."

            return render_template('result.html', prediction=prediction, city=city.title())
        except Exception as e:
            return render_template('windex.html', prediction=f"Error fetching data: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
