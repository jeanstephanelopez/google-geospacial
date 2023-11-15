from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import googlemaps
import os
import requests
from io import StringIO

app = Flask(__name__)

# Fetch and load the CSV data from the URL
csv_url = 'https://raw.githubusercontent.com/jeanstephanelopez/google-geospacial/main/electricity_impact_hurricanes.csv'
csv_data = requests.get(csv_url).text
hurricane_data = pd.read_csv(StringIO(csv_data))
hurricane_data['distance_to_power_source'] = hurricane_data['distance_to_power_source'].apply(lambda x: sum(map(int, x.split('-'))) / 2)

# Initialize Google Maps API client
api_key = os.getenv('GOOGLE_MAPS_API_KEY')
if not api_key:
    raise ValueError("No API key found. Set the GOOGLE_MAPS_API_KEY environment variable.")
gmaps = googlemaps.Client(key=api_key)

# Function to find the nearest power source
def find_nearest_power_source(lat, lng):
    power_sources = ["Power Plant", "Power Station", "nuclear power plant", "coal power plant", "hydroelectric power plant", "solar power plant"]
    nearest_distance = float('inf')
    nearest_source_type = None

    for source in power_sources:
        places_result = gmaps.places_nearby(location=(lat, lng), keyword=source, radius=50000)
        for place in places_result.get('results', []):
            place_location = place['geometry']['location']
            distance_result = gmaps.distance_matrix(origins=(lat, lng), destinations=place_location)
            distance_miles = distance_result['rows'][0]['elements'][0]['distance']['value'] / 1609.34
            if distance_miles < nearest_distance:
                nearest_distance = distance_miles
                nearest_source_type = source

    return nearest_distance, nearest_source_type

# Model training
X = hurricane_data[['hurricane_category', 'distance_to_power_source']]
y = hurricane_data['chance_of_losing_electricity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    lat = data['latitude']
    lng = data['longitude']
    hurricane_category = data['hurricane_category']

    predicted_chance, distance, source_type = make_prediction(hurricane_category, lat, lng)
    response = {
        'predicted_chance_of_losing_electricity': f'{predicted_chance:.2f}%',
        'reason': f'The nearest power source ({source_type}) is {distance:.2f} miles away.'
    }
    return jsonify(response)

def make_prediction(hurricane_category, lat, lng):
    distance, source_type = find_nearest_power_source(lat, lng)
    prediction_input = pd.DataFrame([[hurricane_category, distance]], 
                                    columns=['hurricane_category', 'distance_to_power_source'])
    chance_of_losing_electricity = model.predict(prediction_input)
    return chance_of_losing_electricity[0], distance, source_type

@app.route('/test', methods=['GET'])
def test():
    example_lat = 29.7604
    example_lng = -95.3698
    example_hurricane_category = 3

    predicted_chance, distance, source_type = make_prediction(example_hurricane_category, example_lat, example_lng)
    response = {
        'predicted_chance_of_losing_electricity': f'{predicted_chance:.2f}%',
        'reason': f'The nearest power source ({source_type}) is {distance:.2f} miles away.'
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
