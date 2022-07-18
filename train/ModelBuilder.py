from math import ceil
import timeit

from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

COLOR_CODES = {
    # RGB
    "1": np.array([.18, .31, .31]),
    "2": np.array([.50, 0, 0]),
    "3": np.array([0, .39, 0]),
    "4": np.array([.74, .72, .42]),
    "5": np.array([0, 0, .50]),
    "6": np.array([1.0, 0, 0]),
    "7": np.array([1.0, .65, 0]),
    "8": np.array([1.0, 1.0, 0]),
    "9": np.array([0, 1.0, 0]),
    "10": np.array([0, .98, .60]),
    "11": np.array([0, 0, 1.0]),
    "12": np.array([.85, .44, .84]),
    "13": np.array([1.0, 0, 1.0]),
    "14": np.array([.12, .56, 1.0]),
    "15": np.array([.69, .93, .93]),
    "16": np.array([1.0, .71, .76])
}


class NeuralNet:
    def __init__(self, hparams: dict, data: dict) -> None:
        self.layout = hparams['hidden_layers']
        self.activation = hparams['activation']
        self.optimizer = hparams['optimizer']
        self.lr = hparams['learning_rate']
        self.batch_size = hparams['batch_size']
        self.n_epochs = hparams['n_epochs']

        self.dataset = data['dataset']
        self.origin_data_h = data['data_dim']["origin"][0]
        self.origin_data_w = data['data_dim']["origin"][1]
        self.input_dim = data['data_dim']['n_bands']
        self.output_dim = data['data_dim']['n_classes']
        self.X_all = data['X_all']
        self.y_all = data['y_all']
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.model = None

        self.accuracy = 0.0
        self.training_time = 0.0

    def build(self):
        if len(self.layout) < 1:
            self.layout = [256]
            print(
                f"User didn't input hidden layer, thus will be using system default :{self.layout}")

        optimizer_dict = {
            'adam': keras.optimizers.Adam(learning_rate=self.lr),
            'adamax': keras.optimizers.Adamax(learning_rate=self.lr),
            'ftrl': keras.optimizers.Ftrl(learning_rate=self.lr),
            'nadam': keras.optimizers.Nadam(learning_rate=self.lr),
            'rms_prop': keras.optimizers.RMSprop(learning_rate=self.lr),
            'sgd': keras.optimizers.SGD(learning_rate=self.lr),
        }

        self.model = Sequential()

        for neurons in self.layout:
            self.model.add(Dense(neurons, activation=self.activation))

        self.model.add(Dense(self.output_dim))
        criterion = SparseCategoricalCrossentropy(from_logits=True)
        metric = ['accuracy']

        opt = optimizer_dict[self.optimizer]
        self.model.build(input_shape=(
            (self.origin_data_h * self.origin_data_w), self.input_dim))
        self.model.compile(optimizer=opt, loss=criterion, metrics=metric)

        print(self.model.summary())

    def train(self):
        # Setting up callbacks
        #auto_patience = int(ceil(0.05 * self.n_epochs))
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        callbacks = [early_stop]

        if not self.model:
            print(f"Model does not exist!")
            return None

        print(f"The size of training data: {self.X_train.shape}")
        print(f"The size of testing data: {self.X_test.shape}")

        start = timeit.default_timer()
        self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            validation_split=0.2,
            epochs=self.n_epochs,
            callbacks=callbacks
        )
        stop = timeit.default_timer()
        self.training_time = stop - start
        return self.model

    def evaluate(self):
        if not self.model:
            return f"Model is not found!"

        scores = self.model.evaluate(self.X_test, self.y_test)
        self.accuracy = scores[1] * 100

    def get_results(self):
        return {"acc": self.accuracy, "train_time": self.training_time}


class ML:
    def __init__(self, hparams: dict, data: dict) -> None:
        self.ml_model = hparams['ml_model']

        self.dataset = data['dataset']
        self.origin_data_h = data['data_dim']["origin"][0]
        self.origin_data_w = data['data_dim']["origin"][1]
        self.X_all = data['X_all']
        self.y_all = data['y_all']
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.model = None

        if self.ml_model == 'svm':
            # parameter of SVM
            self.c = hparams['C']  # number
            self.kernel = hparams['kernel']  # 'rbf'
            self.gamma = hparams['gamma']  # 'auto'
            self.model = SVC(
                C=self.c, kernel=self.kernel, gamma=self.gamma)

        elif self.ml_model == 'knn':
            # parameter of KNN
            self.n_neighbors = hparams['n_neighbors']  # number
            self.weights = hparams['weights']  # 'uniform'
            self.metric = hparams['metric']  # 'euclidean'

            if self.metric == 'euclidean':
                self.model = KNeighborsClassifier(
                    n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric, p=2)

            elif self.metric == 'manhattan':
                self.model = KNeighborsClassifier(
                    n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric, p=1)

        self.accuracy = 0.0
        self.training_time = 0.0

    def build(self):
        pass

    def train(self):
        """ Train model
        """
        start = timeit.default_timer()
        self.model.fit(self.X_train, self.y_train)
        stop = timeit.default_timer()
        self.training_time = stop - start
        return self.model

    def evaluate(self):
        """Output accuracy of your model (test on testing data)
        """
        self.accuracy = self.model.score(self.X_test, self.y_test)

    def get_results(self):
        return {"acc": self.accuracy, "train_time": self.training_time}


class SaveResult:
    def __init__(self, hparams: dict, data: dict, algo) -> None:
        self.dataset = data['dataset']
        self.origin_data_h = data['data_dim']["origin"][0]
        self.origin_data_w = data['data_dim']["origin"][1]
        self.X_all = data['X_all']
        self.y_all = data['y_all']
        if algo == 'ml':
            self.algo = hparams['ml_model']  # hparams['ml_model']=svm , knn
        else:  # algo='nn'
            self.algo = algo

    def path_to_save(self):
        pred_file_name = f"./results/{self.dataset}_{self.algo}_pred.png"
        gt_file_name = f"./results/{self.dataset}_{self.algo}_gt.png"
        return pred_file_name, gt_file_name

    def predict_processing(self, model):
        preds = 0
        if self.algo == 'nn':
            # Prediction
            preds_logits = model.predict(self.X_all)
            # The output is logits (e.g., [0.7, 0.2, 0.1]), thus we need to get the
            # highest logit as the classification results
            preds = np.argmax(preds_logits, axis=1)
        else:  # self.algo == 'svm' or 'knn'
            preds = model.predict(self.X_all)

        # The original label is from 1 to 16, which is not acceptable to tensorflow.
        # Thus we subtracted 1 on label when training. Here, we simply add it back.
        preds_corrected = np.add(preds, 1)
        return preds_corrected

    def save_predict(self, preds_corrected, pred_file_name):
        y_all = self.y_all.reshape(-1, 1)
        # Plot all the predictions for different classes.
        largest_classes = np.max(np.unique(y_all).tolist())
        all_classes = [i for i in range(1, largest_classes+1)]
        # preds_colored is the prediction image that combines every classes
        preds_colored = np.zeros((self.origin_data_h * self.origin_data_w, 3))
        # Generate figures for every class from 1 to 16
        for cur_class in all_classes:
            cur_class_wise_img = np.zeros(
                (self.origin_data_h * self.origin_data_w, 3))
            cur_color = COLOR_CODES[str(cur_class)]  # Current class's color
            # Loop over every pixel
            for idx, pixel in enumerate(preds_corrected.tolist()):
                if pixel == cur_class:
                    if y_all[idx] == 0:
                        continue  # Ignore background
                    cur_class_wise_img[idx] = cur_color
                    preds_colored[idx] = cur_color
                else:
                    cur_class_wise_img[idx] = np.array([0, 0, 0])

            # Fix the order issue caused by the file name, e.g., (c1, c10, c11)
            if cur_class < 10:
                class_wise_img_path = f"./results/{self.dataset}_{self.algo}_c0{cur_class}.png"
            else:
                class_wise_img_path = f"./results/{self.dataset}_{self.algo}_c{cur_class}.png"

            # Save class-wise predictions
            plt.axis('off')
            plt.imshow(cur_class_wise_img.reshape(
                self.origin_data_h, self.origin_data_w, 3))
            plt.savefig(class_wise_img_path, bbox_inches='tight', pad_inches=0)

        # Save the full prediction on the whole images
        plt.axis('off')
        plt.imshow(preds_colored.reshape(
            self.origin_data_h, self.origin_data_w, 3))
        plt.savefig(pred_file_name, bbox_inches='tight', pad_inches=0)

    def save_gt(self, gt_file_name):
        # Below we visualize the ground truth
        # Map the colors in ground truth to ours
        gt_colored = np.zeros((self.origin_data_h * self.origin_data_w, 3))
        y_data = self.y_all.reshape(-1, )
        for idx, pixel in enumerate(y_data):
            if pixel == 0:
                continue
            else:
                cur_color = COLOR_CODES[str(pixel)]
                gt_colored[idx] = cur_color

        # Save ground truth using our color codes.
        plt.axis('off')
        plt.imshow(gt_colored.reshape(
            self.origin_data_h, self.origin_data_w, 3))
        plt.savefig(gt_file_name, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    M = 20
    for i in range(1, M+1):
        pi = float(np.power(2, M-i)) / float((np.power(2, M) - 1))
        print(pi)
