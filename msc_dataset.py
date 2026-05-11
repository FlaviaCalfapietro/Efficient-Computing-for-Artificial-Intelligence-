from torch.utils.data import Dataset
import os
import torchaudio

class MSCDataset(Dataset):
	def __init__(self, path: str, classes: list[str] = []):
		self.labels = []
		self.data = []
		self.sampling_rates = []
		self.classes = classes

		# check if the dataset directory exists
		if not os.path.exists(path):
			raise FileNotFoundError()

		# check if dataset path is a directory
		if not os.path.isdir(path):
			raise NotADirectoryError()
		
		# get files from path directory
		contents = os.listdir(path)
		
		for content in contents:
			# check if the current file is a .wav
			name, ext = os.path.splitext(content)
			if ext == ".wav":
       
				# extract label from filenam
				label = name.split("_")[0]

				# add to dataset if it is related to a class in the list
				if label in self.classes:
					# Read class id, audio data and sampling rate and store them
					id = self.label_to_int(label)
					x, sampling_rate = torchaudio.load(path + content)
					self.data.append(x)
					self.labels.append(id)
					self.sampling_rates.append(sampling_rate)
     
	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx) -> dict:
		return {
			"x": self.data[idx],
			"sampling_rate": self.sampling_rates[idx],
			"label": self.labels[idx]
		}

	def label_to_int(self, str) -> int:
		return self.classes.index(str)