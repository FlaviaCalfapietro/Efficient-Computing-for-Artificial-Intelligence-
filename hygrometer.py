import argparse
from typing import Tuple
import adafruit_dht
import uuid
import time
from board import D4
import sounddevice as sd
import redis
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio
import torch
import string
import numpy as np


def parse_args() -> argparse.Namespace:
	"""
	Parse command-line arguments.
	
	Returns:
		argparse.Namespace: Parsed arguments containing host, port, user, and password
	"""
	parser = argparse.ArgumentParser(
		description="Connect to Redis Cloud using provided credentials."
	)

	parser.add_argument("--host", type=str, required=True, help="Redis Cloud host.")
	parser.add_argument("--port", type=int, required=True, help="Redis Cloud port.")
	parser.add_argument("--user", type=str, required=True, help="Redis Cloud username.")
	parser.add_argument("--password", type=str, required=True, help="Redis Cloud password.")
	return parser.parse_args()


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
			temperature = dht_device.temperature
			humidity = dht_device.humidity
			return temperature, humidity, timestamp_ms
		except Exception as e:
			print(f"Sensor failure: {e}")
			try:
				dht_device.exit()
			except Exception:
				pass
			dht_device = adafruit_dht.DHT11(D4)
		attempts -= 1

	raise Exception(f"Cannot read temperature: {max_attempts} attempts failed")

def get_redis_client_instance(
	temp_timeseries_name: str, 
	humidity_timeseries_name: str
) -> redis.Redis:
	"""
	Create and configure a Redis client with TimeSeries support.
	
	Args:
		temp_timeseries_name (str): Name for the temperature time series key
		humidity_timeseries_name (str): Name for the humidity time series key
		
	Returns:
		redis.Redis: Configured Redis client instance with time series initialized
	"""
	args = parse_args()
	redis_client = redis.Redis(
		host=args.host,
		port=args.port,
		username=args.user,
		password=args.password,
	)

	try:
		redis_client.ts().create(temp_timeseries_name)
	except redis.ResponseError:
		pass

	try:
		redis_client.ts().create(humidity_timeseries_name)
	except redis.ResponseError:
		pass

	return redis_client


def run_temperature_loop():
	"""
	Main loop that continuously monitors audio for voice commands and collects sensor data.
	
	Voice commands:
		- "up": Enable data collection
		- "stop": Disable data collection
		
	When enabled, collects temperature and humidity data every 5 seconds and stores in Redis.
	"""
 
	# get mac address and generate timeseries names
	mac_address = hex(uuid.getnode())
	temp_timeseries_name = f"temperature_{mac_address}"
	humidity_timeseries_name = f"humidity_{mac_address}"
	print(temp_timeseries_name)
	print(humidity_timeseries_name)

	# Attach the sensor
	dht_device = adafruit_dht.DHT11(D4)

	# Connect to redis cloud
	redis_client = get_redis_client_instance(temp_timeseries_name, humidity_timeseries_name)

	# Load model and processor
	model_name = "openai/whisper-tiny.en"
	processor = WhisperProcessor.from_pretrained(model_name)
	model = WhisperForConditionalGeneration.from_pretrained(model_name)

	# initialize the resampler from 48khz to 16khz
	samplerate = 48_000
	resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
 
	system_state = False

	# microphone callback (append the data to the buffer)
	def callback(indata, frames, callbacktime, status):
		nonlocal system_state
		## Read audio buffer content
		raw_chunk = indata.copy()

		# transform it into a pytorch tensor
		x = torch.tensor(raw_chunk, dtype=torch.int16)

		# change data layout from channel-last to channel-first
		x = x.transpose(0, 1).contiguous()

		# Transorm to float32 tensor and normalize the waveform values to the range [-1, 1]
		x = x.to(torch.float32) / 32768.0

		# Resample from 48khz to 16khz
		x = resampler(x)

		# get mono audio signal
		if x.shape[0] == 1:
			x_mono = x.squeeze(0)
		else:
			x_mono = x.mean(dim=0)

		# feed the preprocess data to the model
		input_features = processor(x_mono, sampling_rate=16000, return_tensors="pt").input_features
		predicted_ids = model.generate(input_features)
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

		# remove spaces and punctuation
		transcription = transcription.strip().translate(str.maketrans("", "", string.punctuation)).lower()

		# "up" keyword: "ENABLE" the system, "down" keyword "DISABLE" the system
		if "up" in transcription:
			print(f"ENABLED data collection started at {time.time()}")
			system_state = True
		elif "stop" in transcription:
			print(f"DISABLED data collection started at {time.time()}")
			system_state = False

	with sd.InputStream(
		device=1,
		channels=1,
		dtype="int16",
		samplerate=samplerate,
		callback=callback,
		blocksize=samplerate * 1,
	):
		print("System started. Say 'up' to enable or 'stop' to disable.")
		while True:
			if system_state:
				temperature, humidity, timestamp_ms = collect_temperature(dht_device)
				redis_client.ts().add(temp_timeseries_name, timestamp_ms, temperature)
				redis_client.ts().add(humidity_timeseries_name, timestamp_ms, humidity)
				print(f"Sent to Redis: T={temperature:.1f}°C, H={humidity:.1f}% at {timestamp_ms}")

			time.sleep(5)

if __name__ == "__main__":
	run_temperature_loop()