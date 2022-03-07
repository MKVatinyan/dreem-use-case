import json
from pathlib import Path

import h5py
import padasip as pa
from torch.utils.data import Dataset

from helpers import *


class DreemDataset(Dataset):

    def __init__(self, records_list, h5_folder, annotation_folder, processed_data_folder):
        """
        Load row record data, process it and split it to 30 sec data points that will be saved in
        processed_data_folder
        :param records_list: List of record identifiers
        :param h5_folder: folder containing input h5 files
        :param annotation_folder: folder containing hypnograms associated to records in h5_folder
        :param processed_data_folder: folder where 30sec data will be saved as h5 files
        """
        self.processed_data_folder = Path(processed_data_folder)

        # Stop init here if processed data folder already exists and is not empty
        if self.processed_data_folder.is_dir() and any(self.processed_data_folder.iterdir()):
            return

        processed_samples = []
        for record in records_list:
            processed_samples += self.get_processed_samples_for_record(record, h5_folder, annotation_folder)

        self.processed_data_folder.mkdir(exist_ok=True, parents=True)
        self.save_processed_data(processed_samples, self.processed_data_folder)

    def __getitem__(self, idx):
        """
        Return 30sec point array of data and associated label (can be None).
        """
        h5_data_point = h5py.File(Path(self.processed_data_folder, f'{idx}.h5'), 'r')
        label = h5_data_point['label'][()] if 'label' in h5_data_point.keys() else None
        return h5_data_point['data'][:], label

    def __len__(self):
        return len(list(self.processed_data_folder.glob("*.h5")))

    def get_processed_samples_for_record(self, record, h5_folder, annotation_folder):
        # Load raw records and hypnograms
        record_data, hypnogram = self._load_raw_data(record, h5_folder, annotation_folder)

        # Preprocess signals and transform into 30sec data points
        cleaned_record_data = self.process_record_data(record_data)

        # Split night data into data points
        samples = []
        start = 0
        point_idx = 0
        while start < len(cleaned_record_data[0, :]) - ACCELEROMETER_FREQUENCY * SECONDS_IN_POINT:
            point_label = SLEEP_STAGE_ENCODING[hypnogram[point_idx]] if hypnogram is not None else None
            point_data = cleaned_record_data[:, start:start + ACCELEROMETER_FREQUENCY * SECONDS_IN_POINT]

            # Standardize each segment
            point_data_mean = np.mean(point_data, axis=1).reshape(-1, 1)
            point_data_std = np.std(point_data, axis=1).reshape(-1, 1)
            point_data = (point_data - point_data_mean) / point_data_std
            samples.append((point_data, point_label))
            point_idx += 1
            start = point_idx * ACCELEROMETER_FREQUENCY * SECONDS_IN_POINT

        return samples

    def process_record_data(self, record_data):
        """
        Apply processing to a h5 file or numpy array
        """
        # Process accelerometer data
        accelerometer_data = self.process_accelerometer_data(record_data)

        # Process EEG data
        eeg_data = self.process_eeg_data(record_data)

        # Remove movement artifacts from EEG using accelerometer data
        cleaned_eeg_data = self.remove_movement_artifacts(eeg_data, accelerometer_data)

        return cleaned_eeg_data

    @staticmethod
    def _load_raw_data(record, h5_folder, annotation_folder=None):
        """

        :param record:
        :param h5_folder:
        :param annotation_folder:
        :return:
        """
        record_data = h5py.File(h5_folder / f"{record}.h5", 'r')
        if annotation_folder is None:
            hypnogram = None
        else:
            with open(annotation_folder / f"{record}.json", 'r') as hypnogram_file:
                hypnogram = json.load(hypnogram_file)
        return record_data, hypnogram

    @staticmethod
    def save_processed_data(samples, processed_data_folder):
        for i, point in enumerate(samples):
            with h5py.File(Path(processed_data_folder, f'{i}.h5'), 'w') as hf:
                hf.create_dataset('data', data=point[0])
                if point[1] is not None:
                    hf.create_dataset('label', data=point[1])

    @staticmethod
    def process_accelerometer_data(record_data):
        raw_data = [record_data[c][:] for c in ACCELEROMETER_CHANNELS]
        filtered_data = [
            butter_bandpass_filter(axis_data, 0.1, 0.5, ACCELEROMETER_FREQUENCY, order=2) for axis_data in raw_data
        ]
        padded_filtered_data = []
        for signal in filtered_data:
            signal_length = len(signal)
            new_signal_length = (signal_length // (ACCELEROMETER_FREQUENCY * 30) + 1) * ACCELEROMETER_FREQUENCY * 30
            new_signal = np.pad(signal, (0, new_signal_length-signal_length), 'constant', constant_values=0)
            padded_filtered_data.append(new_signal)
        return padded_filtered_data

    @staticmethod
    def process_eeg_data(record_data):
        raw_eeg_data = [record_data[c][:] for c in EEG_CHANNELS]
        filtered_eeg_data = [
            clip_and_scale(butter_bandpass_filter(axis_data, 0.5, 35, EEG_FREQUENCY, order=1))
            for axis_data in raw_eeg_data
        ]

        padded_filtered_data = []
        for signal in filtered_eeg_data:
            signal_length = len(signal)
            new_signal_length = (signal_length // (EEG_FREQUENCY * 30) + 1) * EEG_FREQUENCY * 30
            new_signal = np.pad(signal, (0, new_signal_length - signal_length), 'constant', constant_values=0)
            padded_filtered_data.append(new_signal)

        # Downsample with averaging to have same frequency as accelerometer
        downsampled_filtered_eeg_data = np.array([
            np.mean(x.reshape(-1, EEG_FREQUENCY // ACCELEROMETER_FREQUENCY), axis=1) for x in padded_filtered_data
        ])
        return downsampled_filtered_eeg_data

    @staticmethod
    def remove_movement_artifacts(eeg_data, accelerometer_data):
        accelerometer_activity = np.sqrt(np.sum(np.square(np.array(accelerometer_data)), axis=0))
        cleaned_eeg = np.empty((len(EEG_CHANNELS), eeg_data.shape[1]))

        adaptive_filter_size = ACCELEROMETER_FREQUENCY * 2  # 2sec

        # Accelerometer signal, and downsampled eeg signal
        x_acc = np.array([accelerometer_activity]).repeat(adaptive_filter_size, axis=0).T
        for channel in range(len(EEG_CHANNELS)):
            s_noisy = cleaned_eeg[channel, :]
            f = pa.filters.FilterNLMS(n=adaptive_filter_size, mu=0.1, w="random")
            artifacts, s_clean, w_distortion = f.run(s_noisy, x_acc)
            cleaned_eeg[channel, :] = s_clean

        return cleaned_eeg
