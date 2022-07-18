import sys
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from PyQt5 import *
from PyQt5.QtWidgets import*
from PyQt5.QtCore import *

from train.ServiceController import ServiceController
import json
import os
import glob

menu_widget = uic.loadUiType("menu_widget.ui")[0]
tutorial_widget = uic.loadUiType("tutorial_widget.ui")[0]
training_widget = uic.loadUiType("Training_widget.ui")[0]
testing_widget = uic.loadUiType("testing_widget.ui")[0]


class MenuWidget(QMainWindow, menu_widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.tutorial_button.clicked.connect(self.tutorialbuttonfunc)
        self.training_button.clicked.connect(self.trainingbuttonfunc)
        self.testing_button.clicked.connect(self.testingbuttonfunc)

    def tutorialbuttonfunc(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

    def trainingbuttonfunc(self):
        widget.setCurrentIndex(widget.currentIndex()+2)

    def testingbuttonfunc(self):
        path_dir = './results'
        file_list = os.listdir(path_dir)
        if not file_list:
            QMessageBox.information(self, 'ok', 'Please train your model')
        else:
            widget.setCurrentIndex(widget.currentIndex()+3)


class TutorialWidget(QMainWindow, tutorial_widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.whatis_next.clicked.connect(self.whatisnextfunc)
        self.howto_next.clicked.connect(self.howtonextfunc)
        self.application_next.clicked.connect(self.applicationnextfunc)

        self.whatml_next.clicked.connect(self.whatmlnextfunc)
        self.ml_appli_next.clicked.connect(self.mlappli_nextfunc)

        self.actionhome.triggered.connect(self.toolbar_clicked1)
        self.actiontutorial.triggered.connect(self.toolbar_clicked2)
        self.actiontrain.triggered.connect(self.toolbar_clicked3)
        self.actiontest.triggered.connect(self.toolbar_clicked4)

    def toolbar_clicked1(self):
        widget.setCurrentIndex(0)

    def toolbar_clicked2(self):
        widget.setCurrentIndex(1)

    def toolbar_clicked3(self):
        widget.setCurrentIndex(2)

    def toolbar_clicked4(self):
        path_dir = './results'
        file_list = os.listdir(path_dir)
        if not file_list:
            QMessageBox.information(self, 'ok', 'Please train your model')
        else:
            widget.setCurrentIndex(3)

    def whatisnextfunc(self):
        self.tabWidget_HSI.setCurrentIndex(1)

    def howtonextfunc(self):
        self.tabWidget_HSI.setCurrentIndex(2)

    def applicationnextfunc(self):
        self.tabWidget.setCurrentIndex(1)

    def whatmlnextfunc(self):
        self.tabWidget_ML.setCurrentIndex(1)

    def mlappli_nextfunc(self):
        widget.setCurrentIndex(widget.currentIndex()-1)


class TrainingWidget(QMainWindow, training_widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.dataset_next_button.setDisabled(True)
        self.dataset_next_button.clicked.connect(self.datasetnextbuttonfunc)

        self.groupBox_2.setDisabled(True)
        self.num_band.mousePressEvent = self.enablegroupbox

        self.indian_checkbox.toggled.connect(self.changeTitle)
        self.pavia_checkbox.toggled.connect(self.changeTitle)
        self.salinas_checkbox.toggled.connect(self.changeTitle)

        self.svm_checkbox.toggled.connect(self.changeTitle)
        self.knn_checkbox.toggled.connect(self.changeTitle)
        self.nn_checkbox.toggled.connect(self.changeTitle)

        self.pca_checkbox1.toggled.connect(self.changeTitle)
        self.ica_checkbox1.toggled.connect(self.changeTitle)
        self.non_checkbox1.toggled.connect(self.changeTitle)

        self.taining_next_button.clicked.connect(self.tainingnextbuttonfunc)
        self.taining_start_button.clicked.connect(self.tainingstartbuttonfunc)

        self.groupBox_4.setDisabled(True)
        self.groupBox_5.setDisabled(True)
        self.groupBox_6.setDisabled(True)

        groupcheck = QButtonGroup(self)
        groupcheck.addButton(self.svm_checkbox)
        groupcheck.addButton(self.knn_checkbox)
        groupcheck.addButton(self.nn_checkbox)

        groupcheck2 = QButtonGroup(self)
        groupcheck2.addButton(self.indian_checkbox)
        groupcheck2.addButton(self.pavia_checkbox)
        groupcheck2.addButton(self.salinas_checkbox)

        groupcheck3 = QButtonGroup(self)
        groupcheck3.addButton(self.pca_checkbox1)
        groupcheck3.addButton(self.ica_checkbox1)
        groupcheck3.addButton(self.non_checkbox1)

        self.actionhome.triggered.connect(self.toolbar_clicked1)
        self.actiontutorial.triggered.connect(self.toolbar_clicked2)
        self.actiontrain.triggered.connect(self.toolbar_clicked3)
        self.actiontest.triggered.connect(self.toolbar_clicked4)

        self.num_band.setInputMask("000")
        self.tabWidget.setTabEnabled(1, False)
        self.taining_start_button.setDisabled(True)

        self.c_editor.mousePressEvent = self.clear_c_editor
        self.neighbors.mousePressEvent = self.clear_neighbors
        self.hidden_layer_editor.mousePressEvent = self.clear_hidden_layer_editor
        self.learning_rate_editor.mousePressEvent = self.clear_learning_rate_editor

    def clear_c_editor(self, event):
        self.c_editor.clear()

    def clear_neighbors(self, event):
        self.neighbors.clear()

    def clear_hidden_layer_editor(self, event):
        self.hidden_layer_editor.clear()

    def clear_learning_rate_editor(self, event):
        self.learning_rate_editor.clear()

    def toolbar_clicked1(self):
        widget.setCurrentIndex(0)

    def toolbar_clicked2(self):
        widget.setCurrentIndex(1)

    def toolbar_clicked3(self):
        widget.setCurrentIndex(2)

    def toolbar_clicked4(self):
        path_dir = './results'
        file_list = os.listdir(path_dir)
        if not file_list:
            QMessageBox.information(self, 'ok', 'Please train your model')
        else:
            widget.setCurrentIndex(3)

    def datasetnextbuttonfunc(self):
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget.setTabEnabled(1, True)

    def enablegroupbox(self, event):
        self.groupBox_2.setDisabled(False)
        self.num_band.clear()

    def changeTitle(self):
        self.dataset = ''
        self.dr = 'pca'
        self.algo = ''
        if self.indian_checkbox.isChecked():
            self.dataset_next_button.setDisabled(False)
            self.dataset = "indianpines"
            self.choosed_data.setText('indian pines')
            pixmap = QPixmap("./image/indian pine.jpg")
            self.select_dataset.setPixmap(QPixmap(pixmap))

        if self.pavia_checkbox.isChecked():
            self.dataset_next_button.setDisabled(False)
            self.dataset = 'pavia'
            self.choosed_data.setText('Pavai University')
            pixmap = QPixmap("./image/pavia.jpg")
            self.select_dataset.setPixmap(QPixmap(pixmap))

        if self.salinas_checkbox.isChecked():
            self.dataset_next_button.setDisabled(False)
            self.dataset = 'salinas'
            self.choosed_data.setText('Salinas')
            pixmap = QPixmap("./image/Salinas.jpg")
            self.select_dataset.setPixmap(QPixmap(pixmap))

        if self.svm_checkbox.isChecked():
            self.groupBox_4.setDisabled(False)
            self.groupBox_5.setDisabled(True)
            self.groupBox_6.setDisabled(True)
            self.taining_start_button.setDisabled(True)
            self.c_editor.clear()
            self.algo = 'svm'

        if self.knn_checkbox.isChecked():
            self.groupBox_4.setDisabled(True)
            self.groupBox_5.setDisabled(False)
            self.groupBox_6.setDisabled(True)
            self.taining_start_button.setDisabled(True)
            self.neighbors.clear()
            self.algo = 'knn'

        if self.nn_checkbox.isChecked():
            self.groupBox_4.setDisabled(True)
            self.groupBox_5.setDisabled(True)
            self.groupBox_6.setDisabled(False)
            self.taining_start_button.setDisabled(True)
            self.learning_rate_editor.clear()
            self.hidden_layer_editor.clear()
            self.algo = 'nn'

        if self.pca_checkbox1.isChecked():
            self.dr = 'pca'
            self.num_band.setDisabled(False)

        if self.ica_checkbox1.isChecked():
            self.dr = 'ica'
            self.num_band.setDisabled(False)

        if self.non_checkbox1.isChecked():
            self.dr = 'none'
            self.num_band.setDisabled(True)
            self.num_band.setText('0')
            self.groupBox_2.setDisabled(False)

    def tainingnextbuttonfunc(self):
        if self.algo == 'nn':
            try:
                self.client_data = {
                    "dataset": self.dataset,
                    "dim_reduce": [self.dr, int(self.num_band.text())],
                    "algo": self.algo,
                    "hparams": {
                        'test_ratio': float(self.NN_split_2.value()/100),
                        'hidden_layers': self.hidden_layer_editor.text().strip('][').split(','),
                        'batch_size': int(self.batchsize.currentText()),
                        'n_epochs': int(self.epochs.currentText()),
                        'optimizer': self.optimizer.currentText(),
                        'activation': self.activation.currentText(),
                        'learning_rate': float(self.learning_rate_editor.text())
                    }
                }

                self.client_data_view = "dataset: {}\ndim_reduce: {}\nalgo: {}\ntest_ratio: {}\nhidden_layers: {}\nbatch_size: {}\nn_epochs: {}\noptimizer: {}\nactivation: {}\nlearning_rate: {}".format(self.dataset,
                                                                                                                                                                                                          [self.dr, int(self.num_band.text())], self.algo, float(self.NN_split_2.value(
                                                                                                                                                                                                          )/100),  self.hidden_layer_editor.text().strip('][').split(','), int(self.batchsize.currentText()), int(self.epochs.currentText()),
                                                                                                                                                                                                          self.optimizer.currentText(), self.activation.currentText(), float(self.learning_rate_editor.text()))
                buttonable = 0
                if float(self.learning_rate_editor.text()) > 10 or float(self.learning_rate_editor.text()) < 0.0001:
                    self.client_data_view = 'Please rewrite your learning rate between 0.0001~10 '
                    buttonable = 1
                    for i in self.hidden_layer_editor.text().split(','):
                        if not i.isnumeric():
                            self.client_data_view += '\n\nPlease rewrite your hidden_layers as 256, 128, 64'
                            buttonable = 1
                        elif int(i) > 1024 or int(i) < 0:
                            self.client_data_view += '\n\nPlease rewrite your hidden_layers as 256, 128, 64\n\nNumber of hidden layers range is 0~1024'
                            buttonable = 1
                        self.textBrowser_2.clear()
                        self.textBrowser_2.append(self.client_data_view)

                else:
                    for i in self.hidden_layer_editor.text().split(','):
                        if not i.isnumeric():
                            self.client_data_view = 'Please rewrite your hidden_layers as 256, 128, 64'
                            buttonable = 1
                        elif int(i) > 1024 or int(i) < 0:
                            self.client_data_view = 'Please rewrite your hidden_layers as 256, 128, 64\n\nNumber of hidden layers range is 1~1024'
                            buttonable = 1
                        self.textBrowser_2.clear()
                        self.textBrowser_2.append(self.client_data_view)

                if buttonable == 0:
                    self.taining_start_button.setDisabled(False)
                    self.textBrowser_2.clear()
                    self.textBrowser_2.append(self.client_data_view)
                print(self.client_data)
            except:
                self.textBrowser_2.clear()
                self.textBrowser_2.append("Fail:\nPlease check your parameter")

        elif self.algo == 'svm':
            try:
                self.client_data = {
                    "dataset": self.dataset,
                    "dim_reduce": [self.dr, int(self.num_band.text())],
                    "algo": 'ml',
                    "hparams": {
                        'test_ratio': float(self.SVM_split.value()/100),
                        'ml_model': self.algo,
                        'C': float(self.c_editor.text()),
                        'gamma': int(self.gamma.currentText()),
                        'kernel': self.kernel.currentText(),
                    }
                }

                self.client_data_view = "dataset: {}\ndim_reduce: {}\nalgo: {}\ntest_ratio: {}\nC: {}\ngamma: {}\nkernel: {}".format(self.dataset,
                                                                                                                                     [self.dr, int(self.num_band.text())], self.algo, float(
                                                                                                                                         self.SVM_split.value()/100), float(self.c_editor.text()),
                                                                                                                                     int(self.gamma.currentText()), self.kernel.currentText(),)
                buttonable = 0
                if float(self.c_editor.text()) < 0.001 or float(self.c_editor.text()) > 100:
                    self.client_data_view = 'range of C is 0.001 ~ 100'
                    buttonable = 1
                    self.textBrowser_2.clear()
                    self.textBrowser_2.append(self.client_data_view)

                if buttonable == 0:
                    self.taining_start_button.setDisabled(False)
                    self.textBrowser_2.clear()
                    self.textBrowser_2.append(self.client_data_view)

                print(self.client_data)
            except:
                self.textBrowser_2.clear()
                self.textBrowser_2.append("Fail:\nPlease check your parameter")

        elif self.algo == 'knn':
            try:
                self.client_data = {
                    "dataset": self.dataset,
                    "dim_reduce": [self.dr, int(self.num_band.text())],
                    "algo": 'ml',
                    "hparams": {
                        'test_ratio': float(self.knn_split_2.value()/100),
                        'ml_model': self.algo,
                        'n_neighbors': int(self.neighbors.text()),
                        'weights': self.weights.currentText(),
                        'metric': self.metrics.currentText(),
                    }
                }

                self.client_data_view = "dataset: {}\ndim_reduce: {}\nalgo: {}\ntest_ratio: {}\nn_neighbor: {}\nweights: {}\nmetric: {}".format(self.dataset,
                                                                                                                                                [self.dr, int(self.num_band.text())], self.algo, float(
                                                                                                                                                    self.SVM_split.value()/100), int(self.neighbors.text()),
                                                                                                                                                self.weights.currentText(), self.metrics.currentText())
                buttonable = 0
                if int(self.neighbors.text()) < 0 or int(self.neighbors.text()) > 100:
                    self.client_data_view = 'range of n_neighbor is 1 ~ 100'
                    buttonable = 1
                    self.textBrowser_2.clear()
                    self.textBrowser_2.append(self.client_data_view)

                if buttonable == 0:
                    self.taining_start_button.setDisabled(False)
                    self.textBrowser_2.clear()
                    self.textBrowser_2.append(self.client_data_view)
                print(self.client_data)

            except:
                self.textBrowser_2.clear()
                self.textBrowser_2.append("Fail:\nPlease check your parameter")

    def _process_train(self, result):
        self.result = json.loads(result)
        self.textBrowser_2.clear()
        print(self.result)
        print('Done')
        widget.setCurrentIndex(3)

        self.learning_rate_editor.clear()
        self.hidden_layer_editor.clear()
        self.taining_start_button.setDisabled(True)
        self.c_editor.clear()
        self.neighbors.clear()

        path_dir = './results'
        file_list = os.listdir(path_dir)
        for i, file in enumerate(file_list):
            file_list[i] = file.split('_')[2].split('.')[0]

        testingwindow.comboBox.clear()
        testingwindow.comboBox.addItems(list(file_list))
        pixmap = QPixmap("./results/{}_gt.png".format('_'.join(os.listdir('./results')
                                                               [1].split('_')[:2])))
        testingwindow.ground_truth.setPixmap(QPixmap(pixmap))
        pixmap = QPixmap("./results/{}_pred.png".format('_'.join(os.listdir('./results')
                                                                 [1].split('_')[:2])))

        testingwindow.pred.setPixmap(QPixmap(pixmap))
        self.groupBox_2.setDisabled(False)
        self.groupBox.setDisabled(False)
        self.actionhome.setDisabled(False)
        self.actiontutorial.setDisabled(False)
        self.actiontrain.setDisabled(False)
        self.actiontest.setDisabled(False)
        self.tabWidget.setTabEnabled(0, True)
        self.tabWidget.setTabEnabled(1, True)

        self.loading.close()
        self.SVM_split.setValue(0)
        self.knn_split_2.setValue(0)
        self.NN_split_2.setValue(0)
        testingwindow.textBrowser.clear()
        testingwindow.textBrowser.append("Train success!!!\n\nAccuracy: {} %\ntrain_time: {} sec\n\n{}".format(
            round(self.result['acc'], 3), round(self.result['train_time'], 3), self.client_data_view))

    def tainingstartbuttonfunc(self):
        [os.remove(f) for f in glob.glob("./results/*.png")]
        QMessageBox.information(self, 'ok', 'training start')
        self.groupBox_2.setDisabled(True)
        self.groupBox.setDisabled(True)

        self.actionhome.setDisabled(True)
        self.actiontutorial.setDisabled(True)
        self.actiontrain.setDisabled(True)
        self.actiontest.setDisabled(True)
        self.tabWidget.setTabEnabled(0, False)
        self.tabWidget.setTabEnabled(1, False)

        self.operator = ServiceController(self.client_data)
        self.operator.start()
        self.operator.return_sig.connect(self._process_train)

        self.loading = loading(self)


class loading(QWidget, uic.loadUiType("load.ui")[0]):

    def __init__(self, parent):
        super(loading, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.show()

        self.movie = QMovie('./image/loader.gif', QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)

        self.label.setMovie(self.movie)
        self.movie.start()
        self.setWindowFlags(Qt.FramelessWindowHint)

    def center(self):
        size = self.size()
        ph = self.parent().geometry().height()
        pw = self.parent().geometry().width()
        self.move(int(pw/2 - size.width()/2), int(ph/2 - size.height()/2))


class TestingWidget(QMainWindow, testing_widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.actionhome.triggered.connect(self.toolbar_clicked1)
        self.actiontutorial.triggered.connect(self.toolbar_clicked2)
        self.actiontrain.triggered.connect(self.toolbar_clicked3)
        self.actiontest.triggered.connect(self.toolbar_clicked4)
        self.search.clicked.connect(self.combo_box_select_changed)

    def combo_box_select_changed(self):
        print("./result/{}_{}.png".format('_'.join(os.listdir('./results')
              [1].split('_')[:2]), self.comboBox.currentText()))
        pixmap = QPixmap("./results/{}_{}.png".format('_'.join(os.listdir('./results')
                                                               [1].split('_')[:2]), self.comboBox.currentText()))
        self.pred.setPixmap(QPixmap(pixmap))

    def toolbar_clicked1(self):
        widget.setCurrentIndex(0)

    def toolbar_clicked2(self):
        widget.setCurrentIndex(1)

    def toolbar_clicked3(self):
        widget.setCurrentIndex(2)

    def toolbar_clicked4(self):
        path_dir = './results'
        file_list = os.listdir(path_dir)
        if not file_list:
            QMessageBox.information(self, 'ok', 'Please train your model')
        else:
            widget.setCurrentIndex(3)

    def load(self):
        print("test")
        path_dir = './results'
        file_list = os.listdir(path_dir)
        for i, file in enumerate(file_list):
            file_list[i] = file.split('_')[2].split('.')[0]
        self.comboBox.addItems(list(file_list))


if __name__ == "__main__":
    [os.remove(f) for f in glob.glob("./results/*.png")]

    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()

    menuwindow = MenuWidget()
    tutorialwindow = TutorialWidget()
    traningwindow = TrainingWidget()
    testingwindow = TestingWidget()

    widget.addWidget(menuwindow)
    widget.addWidget(tutorialwindow)
    widget.addWidget(traningwindow)
    widget.addWidget(testingwindow)
    widget.setFixedSize(730, 430)
    widget.show()

    app.exec_()
