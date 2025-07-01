from flask import Flask, render_template, request
import joblib
import requests

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("weather_model.pkl")

# OpenWeather API key
API_KEY = "your_openweather_api_key"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        city = request.form['city']  # Get the city name from user input

        # Validate input
        if not city.strip():
            return render_template('index.html', prediction="Please enter a valid city name.")

        # Fetch current weather data from OpenWeather API
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code != 200:
            return render_template('index.html', prediction="City not found or API error.")

        data = response.json()
        current_temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        # Prepare features for prediction (example: [current_temp, humidity, wind_speed])
        features = [[current_temp, humidity, wind_speed]]

        # Predict the future temperature
        predicted_temp = model.predict(features)[0]
        formatted_predicted_temp = f"{predicted_temp:.2f} °C"

        # Pass results to the template
        return render_template('result.html', 
                               city=city, 
                               current_temp=f"{current_temp} °C", 
                               humidity=f"{humidity}%", 
                               wind_speed=f"{wind_speed} m/s", 
                               predicted_temp=formatted_predicted_temp)

if __name__ == '__main__':
    app.run(debug=True)
