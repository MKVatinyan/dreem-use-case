import numpy as np
from scipy.signal import butter, sosfiltfilt

SECONDS_IN_POINT = 30

EEG_CHANNELS = [f'eeg{i}' for i in range(1, 8)]
EEG_FREQUENCY = 250

ACCELEROMETER_CHANNELS = [f'accelerometer_{x}' for x in ['x', 'y', 'z']]
ACCELEROMETER_FREQUENCY = 50

SLEEP_STAGE_ENCODING = {
    "WAKE": 0,
    "N1": 1,
    "N2": 2,
    "DEEP": 3,
    "REM": 4
}


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Return a bandpass filter on [lowcut,highcut] band
    lowcut: Lower bound of the filter, in hz
    highcut: Higher bound of the filter, in hz
    fs: frequency of the signal, in hz
    order: order of the filter, higher order will result in stronger regularization

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    recommended to use sos output for numerical stability
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Apply a bandpass filtering on [lowcut,highcut] band on the provided data
    data: 1D numpy array
    lowcut: Lower bound of the filter, in hz
    highcut: Higher bound of the filter, in hz
    fs: frequency of the signal, in hz
    order: order of the filter, higher order will result in stronger regularization
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)

    # Apply filter forwards and backwards to remove phase delay ?
    # This double the order (order with sosfiltfilt is like 2xorder with sosfilt)
    y = sosfiltfilt(sos, data)
    return y


def clip_and_scale(data, max_value: float = 300):
    """
    Clip and scale a numpy array to ensure that its value are between -1 and 1.
    data: 1D numpy array
    max_value: value below - max_value or above max_value will be clipped resp to - max_value and to max_value
    """
    return np.clip(np.array(data), -max_value, max_value) / max_value


def get_one_hot_encoding(arr):
    new_arr = np.zeros((len(arr), 5))
    for i, val in enumerate(arr):
        new_arr[i, val] = 1
    return new_arr
