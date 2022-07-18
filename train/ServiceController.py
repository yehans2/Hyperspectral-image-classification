import json

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

from train.Trainer import Trainer


class ServiceController(QtCore.QThread):
    """This class is used to hold the command and data sent from the frontend.
    """
    return_sig = pyqtSignal(str)

    def __init__(self, hparams):
        super().__init__()
        # data: a dict {'algo': str, 'n_components': int} if command is dr.
        self.hparams = hparams

    def run(self):
        # Send command and data
        result = Trainer().main(self.hparams)

        # Emit data to frontend
        self.return_sig.emit(json.dumps(result))


if __name__ == "__main__":
    client_data = {
        "dataset": "indian_pines",
        "dim_reduce": ["pca", 16],  # "pca", "ica" available
        "algo": 'nn',
        "hparams": {
            'test_ratio': 0.2,
            'hidden_layers': [256, 128, 64],
            'batch_size': 32,
            'n_epochs': 10,
            # "adam", "adamax", "nadam", "ftrl","rms_prop","sgd" are available
            'optimizer': 'nadam',
            'activation': 'relu',  # All tensorflow activations are available
            'learning_rate': 0.001
        }
    }
