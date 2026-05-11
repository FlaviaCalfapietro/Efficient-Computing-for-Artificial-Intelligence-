import uuid
import adafruit_dht
from board import D4
from typing import Tuple
import time
import paho.mqtt.client as mqtt
import json
import traceback


def collect_temperature(
	dht_device: adafruit_dht.DHT11, 
	max_attempts: int = 10
) -> Tuple[float, float, int]:
	"""
	Collect temperature and humidity readings from DHT11 sensor.
	
	Args:
		dht_device: DHT11 sensor device instance
		max_attempts (optional): max number of failed attempts permitted, 
  			if the number will be overcomed, it raise an Exception
		
	Returns:
		tuple: (temperature in Celsius, humidity in %, timestamp in milliseconds)
	"""
	attempts = max_attempts
	while attempts > 0:
		try:
			timestamp = time.time()
			timestamp_ms = int(timestamp * 1000)
			temperature = int(dht_device.temperature)
			humidity = int(dht_device.humidity)
			return temperature, humidity, timestamp_ms
		except Exception as e:
			try:
				dht_device.exit()
			except Exception:
				pass
			dht_device = adafruit_dht.DHT11(D4)
		attempts -= 1

	raise Exception(f"Cannot read temperature: {max_attempts} attempts failed")


def main():
	mac_address = hex(uuid.getnode())
	print(mac_address)

	# Attach the sensor
	dht_device = adafruit_dht.DHT11(D4)

	client = mqtt.Client()
	client.connect('broker.emqx.io', 1883)

	while True:
		try:
			temperature, humidity, timestamp_ms = collect_temperature(dht_device)
			message: dict = {
				"timestamp": timestamp_ms,
				"mac_address": mac_address,
				"data": [
					{"name": "temperature", "value": temperature},
					{"name": "humidity", "value": humidity}
				]
			}

			client.publish('s337116', json.dumps(message))
			print(f"Published: {message}")
		except Exception as e:
			print(f"Error collecting or publishing data: {e}")
		time.sleep(5)

if __name__ == "__main__":
	main()