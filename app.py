# app.py
import shutil
from pathlib import Path

import boto3
import torch
from botocore import UNSIGNED
from botocore.client import Config
from flask import Flask, jsonify, abort

from dataset import DreemDataset
from helpers import SLEEP_STAGE_ENCODING
from model import BasicModel


app = Flask(__name__)


@app.get("/hypnogram/<string:record_identifier>")
def get_hypnogram(record_identifier: str):
    """
    Download record if available, transform into 30sec data points using the dataset class, load trained model and
    get predictions and return reconstructed hypnogram

    :return: Hypnogram list of strings
    """
    # Check if record exists
    if record_identifier not in [p.split('.')[0] for p in list_h5()]:
        abort(404, "Requested record not found")

    # Download data into tmp folder
    tmp_folder = Path('data/tmp_h5')
    if tmp_folder.is_dir():
        shutil.rmtree(str(tmp_folder))

    download_h5(f'{record_identifier}.h5', str(tmp_folder))
    print("Downloaded")
    # Read loaded dataset, process and write in another tmp folder
    processed_tmp_folder = Path('data/tmp_h5_processed')
    if processed_tmp_folder.is_dir():
        shutil.rmtree(str(processed_tmp_folder))

    record_list = [record_identifier]
    dataset = DreemDataset(record_list, tmp_folder, None, processed_tmp_folder)
    shutil.rmtree(str(tmp_folder))

    model = BasicModel()
    model.load_state_dict(torch.load('models/saved_model_state_dict.pth'))
    model.eval()

    sleep_stage_decoding = {v: k for k, v in SLEEP_STAGE_ENCODING.items()}
    predicted_hypnogram = []
    for i in range(len(dataset)):
        data = dataset[i]
        input_signals = torch.unsqueeze(torch.tensor(data[0]), 0)
        prediction = model(input_signals.float())
        predicted_sleep_stage = torch.nn.Softmax(dim=1)(prediction).argmax().item()
        predicted_hypnogram.append(sleep_stage_decoding[predicted_sleep_stage])

    # Remove tmp folder
    shutil.rmtree(str(processed_tmp_folder))
    return jsonify(predicted_hypnogram)


def list_h5():
    """
    List the h5 files on the bucket
    :return: List[str]
    """
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_objects = client.list_objects(Bucket='dreem-ml-case-study')["Contents"]
    return [x['Key'] for x in bucket_objects]


def download_h5(h5: str, folder: str):
    """
    Download a h5 file from the aws bucket
    :return: List[str]
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    client.download_file(
        Bucket="dreem-ml-case-study",
        Key=h5,
        Filename=str(folder / h5)
    )
