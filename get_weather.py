"""
title: NOAA Weather & Tides
author: Daniel Saleeb
author_url: https://github.com/dsaleeb
version: 0.1.0
"""

import requests
import datetime

def get_coords(city):
    """Get latitude and longitude for a city using Open-Meteo's geocoder."""
    response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json&countryCode=US")
    if response.ok and response.json().get("results"):
        result = response.json()["results"][0]
        return result["latitude"], result["longitude"]
    return None, None

def get_gridpoint_url(lat, lon):
    """Return forecast URL from NOAA's weather.gov for given lat/lon."""
    headers = {"User-Agent": "NOAAWeatherTide/1.0 (your_email@example.com)"}
    response = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers)
    if response.ok:
        return response.json()["properties"]["forecast"]
    return None

def get_weather_forecast(city):
    """Get 7-period weather forecast from NOAA for a city."""
    lat, lon = get_coords(city)
    if not lat or not lon:
        return "City not found."

    forecast_url = get_gridpoint_url(lat, lon)
    if not forecast_url:
        return "Forecast URL not found from NWS."

    headers = {"User-Agent": "NOAAWeatherTide/1.0 (your_email@example.com)"}
    response = requests.get(forecast_url, headers=headers)
    if not response.ok:
        return "Failed to retrieve forecast."

    periods = response.json()["properties"]["periods"]
    summary = []
    for period in periods[:7]:
        summary.append(f"{period['name']}: {period['shortForecast']}, {period['temperature']}°{period['temperatureUnit']}")

    return "\n".join(summary)

def get_tide_predictions(station_id="9447130"):
    """Get high/low tide predictions for the next 2 days from NOAA CO-OPS."""
    end_date = datetime.datetime.utcnow().date() + datetime.timedelta(days=2)
    start_date = end_date - datetime.timedelta(days=1)

    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "begin_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "station": station_id,
        "product": "predictions",
        "datum": "MLLW",
        "units": "english",
        "time_zone": "lst_ldt",
        "format": "json",
        "interval": "hilo"
    }

    response = requests.get(url, params=params)
    if not response.ok:
        return "Failed to get tide data."

    predictions = response.json().get("predictions", [])
    output = []
    for p in predictions:
        time = datetime.datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
        output.append(f"{time.strftime('%b %d, %I:%M %p')} - {p['type']} tide at {p['v']} ft")

    return "\n".join(output)

# Tools for Open WebUI
class Tools:
    def get_weekly_weather_forecast(self, city: str) -> str:
        """
        Get a 7-period forecast from NOAA for a given city.
        :param city: The name of the city.
        :return: A readable weather forecast summary.
        """
        return get_weather_forecast(city)

    def get_tide_predictions(self, station_id: str = "9447130") -> str:
        """
        Get tide predictions for a NOAA tide station.
        :param station_id: The NOAA station ID (e.g., Seattle = 9447130).
        :return: High/low tide info.
        """
        return get_tide_predictions(station_id)

city="Seattle"
print(get_weather_forecast(city))
