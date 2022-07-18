from train.DataHolder import DataHolder
from train.ModelBuilder import NeuralNet
from train.ModelBuilder import ML
from train.ModelBuilder import SaveResult
#                        +-------------------------------------------------------------------+
#                        |   Trainer.py Class                                                |
#                        |                                  +---------------------------+    |
#  +---------------+     |   +---------------------+        | Build and train the model |    |   Return model and other       +---------------+
#  | Frontend (UI) | ----+-> | DataHolder.py Class | -----> | based on the given        | ---+------------------------------> | Frontend (UI) |
#  +---------------+     |   +---------------------+        | hyperparameters           |    |   statistics to frontend       +---------------+
#                        |   Preprocess the data:           +---------------------------+    |
#                        |   1. Normalization                                                |
#                        |   2. Shuffling                                                    |
#                        |   3. Train-test split                                             |
#                        |   This class holds data,                                          |
#                        |   users can access data                                           |
#                        |   through this class                                              |
#                        |                                                                   |
#                        +-------------------------------------------------------------------+


class Trainer:
    def __init__(self, mode='train') -> None:
        # mode = train / test
        self.mode = mode

    def main(self, received_params: dict):
        """Return model performance based on given parameters

        Args:
            received_params (dict): The format of received_params is shown at the bottom
            of this doc. (client_data)

        Returns:
            result: Accuracy on testing data and the training time.
        """
        if self.mode == 'train':
            dataset = received_params['dataset']
            hparams = received_params['hparams']

            self.data_holder = DataHolder(dataset)
            self.data_holder.main('load_data', None)
            self.data_holder.main(
                'dim_reduce',
                {
                    'dr_algo': received_params['dim_reduce'][0],
                    'n_components': received_params['dim_reduce'][1]
                }
            )
            self.data_holder.main('split', hparams['test_ratio'])
            data = self.data_holder.main('get_data', None)

            if received_params['algo'] == 'nn':
                model_holder = NeuralNet(hparams, data)
            elif received_params['algo'] == 'ml':
                model_holder = ML(hparams, data)

            model_holder.build()
            model = model_holder.train()
            model_holder.evaluate()

            result = model_holder.get_results()
            save_control = SaveResult(hparams, data, received_params['algo'])
            pred_file_name, gt_file_name = save_control.path_to_save()
            preds_corrected = save_control.predict_processing(model)
            save_control.save_predict(preds_corrected, pred_file_name)
            save_control.save_gt(gt_file_name)

            return result


if __name__ == "__main__":
    trainer = Trainer('train')
    # client_data = {
    #     "dataset": "indianpines",
    #     "dim_reduce": ["pca", 16],  # "pca", "ica" available
    #     "algo": 'nn',
    #     "hparams": {
    #         'test_ratio': 0.2,
    #         'hidden_layers': [256, 128, 64],
    #         'batch_size': 32,
    #         'n_epochs': 10,
    #         # "adam", "adamax", "nadam", "ftrl","rms_prop","sgd" are available
    #         'optimizer': 'nadam',
    #         'activation': 'relu',  # All tensorflow activations are available
    #         'learning_rate': 0.001
    #     }
    # }
    # client_data = {
    #     "dataset": "indianpines",
    #     "dim_reduce": ["pca", 16],  # "pca", "ica" available
    #     "algo": 'ml',
    #     "hparams": {
    #         'ml_model': 'svm',
    #         'test_ratio': 0.2,
    #         'C': 0.01,
    #         'kernel': 'linear',
    #         'gamma': 1
    #     }
    # }
    client_data = {
        "dataset": "indianpines",
        "dim_reduce": ["pca", 16],  # "pca", "ica" available
        "algo": 'ml',
        "hparams": {
            'ml_model': 'knn',
            'test_ratio': 0.2,
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski'
        }
    }
    result = trainer.main(client_data)

# All available activations: https://www.tensorflow.org/api_docs/python/tf/keras/activations
