from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

import imutils
import numpy as np
import h5py
import keras.backend as K
import cv2
import json
import os
import random
import utils
import time
import csv
from MetricsHistory import MetricsHistory
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from lenet import LeNet
from lenet2 import LeNet2
from imutils import paths
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TB_ROOT = "C:/tensorboards/"

class BreastCancerDetector:

    def __init__(self):
        self._EPOCHS = None
        self._INIT_LR = None
        self._BS = None
        self._opt = None
        self._loss = None
        self._H = None
        self._model = None


    def config(self, config_filename, config_id):
        with open(config_filename) as json_file:
            configs = json.load(json_file)

        self._EPOCHS = configs["config"][config_id]["epochs"]
        self._INIT_LR = configs["config"][config_id]["init_lr"]
        self._BS = configs["config"][config_id]["batch_size"]
        self._opt = configs["config"][config_id]["opt"]
        self._loss = configs["config"][config_id]["loss"]

    def load_dataset(self, dataset_path):
        # initialize the data and labels
        print("[INFO] loading images...")
        data = []
        labels = []

        # grap the image path and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(dataset_path)))
        random.seed(42)
        random.shuffle(imagePaths)

        # loop over the input images
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            image = img_to_array(image)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]
            label = 1 if label == "benign" else 0
            labels.append(label)

        # scale the raw pixel intensitites to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        return data, labels

    def process_dataset(self, filename):

        width = 128
        height = 128

        data = h5py.File(filename)

        print('[INFO] Loading dataset with keys...')
        for k, v in data.items():
            print(k, v.shape)

        # bg_tissue = data['BG'][:]
        # radius = data['RADIUS'][:]

        # classes = data['CLASS'][:]
        # convert to numerical data
        # class_le = LabelEncoder()
        # class_le.fit(classes)
        # class_info = class_le.transform(classes)
        # labels = np.array(class_info)

        severity = [s.decode('utf-8') for s in data['SEVERITY'][:]]
        severity_le = LabelEncoder()
        severity_le.fit(severity)
        severity_info = severity_le.transform(severity)
        labels = np.array(severity_info) #  0 = B, 1 = N, 2 = M

        scans = data['scan'][:]
        scans = [img_to_array(cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)) for img in scans]

        # scale the pixels intensities
        scans = np.array(scans, dtype="float") / 255.0

        return scans, labels

    def load_model(self, filename):

        self._model = load_model(filename)

    def visualize_data_augmentation(self, img_name, preview_folder):

        # custom = CustomAugmentation(random_crop_size=(50, 50))

        aug = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                vertical_flip=True,
                                samplewise_std_normalization=True,
                                zca_whitening=True,
                                fill_mode='nearest')

        img = cv2.imread(img_name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        if not os.path.exists(preview_folder):
            os.makedirs(preview_folder)

        i = 0
        for batch in aug.flow(x, batch_size=1,
                              save_to_dir=preview_folder, save_prefix='scan', save_format='jpg',):
            i += 1
            if i > 20:
                break

    def preprocessing(self, data, labels, n_classes, rnd_state=None):

        total, n_benign = len(labels), np.count_nonzero(labels)
        n_malignant = total - n_benign
        print('[INFO] Dataset: {} mammographies, {} benign and {} malignant'.format(total, n_benign, n_malignant))

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=rnd_state)

        total_train, n_benign_train = len(trainY), np.count_nonzero(trainY)
        n_malignant_train = total_train - n_benign_train
        print('[INFO] Total {} train mammographies, {} benign, {} malignant'.format(total_train, n_benign_train,
                                                                                    n_malignant_train))

        total_test, n_benign_test = len(testY), np.count_nonzero(testY)
        n_malignant_test = total_test - n_benign_test
        print('[INFO] Total {} test mammographies, {} benign, {} malignant'.format(total_test, n_benign_test,
                                                                                   n_malignant_test))

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=n_classes)
        testY = to_categorical(testY, num_classes=n_classes)

        return trainX, trainY, testX, testY

    def train_evaluate_lenet(self, n_classes, trainX, trainY, testX, testY):

        trainY = to_categorical(trainY, num_classes=n_classes)
        testY = to_categorical(testY, num_classes=n_classes)

        # aug = ImageDataGenerator()

        # custom = CustomAugmentation(random_crop_size=(50, 50))

        aug = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 vertical_flip=True,
                                 #samplewise_std_normalization=True,
                                 #zca_whitening=True,
                                 fill_mode='nearest')

        # initialize the model
        print("[INFO] compiling the model...")
        model = LeNet.build(width=28, height=28, depth=1, classes=n_classes)
        # model = LeNet2.build(width=128, height=128, depth=1, classes=n_classes)

        opt = Adam(lr=self._INIT_LR, decay=self._INIT_LR / self._EPOCHS)

        model.compile(loss=self._loss, optimizer=opt, metrics=["binary_accuracy"])

        print("[INFO] initializing Tensorboard...")

        now = time.strftime("%w-%m-%Y_%M-%S", time.localtime(time.time()))

        tensorboard = TensorBoard(log_dir="{}breast_cancer_mias/training_{}".format(TB_ROOT, now),
                                  histogram_freq=1,
                                  write_graph=True,
                                  write_images=True)

        # csv_logger = CSVLogger('metrics/training_{}.log'.format(now))

        history_log = MetricsHistory()

        # train the network
        print("[INFO] training the network...")

        self._H = model.fit_generator(aug.flow(trainX, trainY, batch_size=self._BS),
                                      validation_data=(testX, testY), steps_per_epoch=len(trainX) // self._BS,
                                      epochs=self._EPOCHS, verbose=2, callbacks=[history_log])

        # metrics = history_log.get_mean_values()

        # print("[INFO] Training: Mean loss: {0:.4} Mean acc: {1:.4}".format(metrics['MLoss'], metrics['MBin_acc']))
        # print("[INFO] Validation: Mean loss: {0:.4} Mean acc: {1:.4}".format(metrics['MVal_loss'], metrics['MVal_bin_acc']))

        return model  # , metrics

    def train(self, dataset_name, saved_model):

        data, labels = self.load_dataset(dataset_name)

        # get the number of classes
        n_classes = len(np.unique(labels))

        trainX, trainY, testX, testY = self.preprocessing(data, labels, n_classes)

        model = self.train_evaluate_lenet(n_classes, trainX, trainY, testX, testY)

        # save the model to disk
        print("[INFO] serializing network...")
        model.save(saved_model)

        return model

    def evaluate_models(self, dataset_name, n_repeats, n_splits):

        data, labels = self.load_dataset(dataset_name)

        # get the number of classes
        n_classes = len(np.unique(labels))

        kfold = KFold(n_splits=n_splits, shuffle=True)

        grand_mean_loss = []
        grand_mean_acc = []

        csv_header = ['Grand_mean_loss', 'Grand_mean_acc']
        csv_file = open("metrics/grand_mean.csv", 'w')

        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        csv_writer.writeheader()
        where_to_save = "models/with_data_augmentation/"

        for i in range(0, n_repeats):
            loss = []
            acc = []
            for j, (train, test) in enumerate(kfold.split(data, labels)):
                print("[INFO] Repeat #{} - Running Fold {}/{}".format(i, (j + 1), n_splits))
                print("[INFO] Training size: {} Testing size: {}".format(len(train), len(test)))
                model = None  # Clearing the NN.
                model = self.train_evaluate_lenet(n_classes, data[train], labels[train], data[test], labels[test])
                loss.append(np.mean(self._H.history["val_loss"]))
                acc.append(np.mean(self._H.history["val_binary_accuracy"]))

                # save the model to disk
                print("[INFO] serializing network...")
                model.save("{}model_run{}_fold{}_{}.h5".format(where_to_save,i, (j + 1), n_splits))

                K.clear_session()  # !!!!
            row = {'Grand_mean_loss': np.mean(loss), 'Grand_mean_acc': np.mean(acc)}
            csv_writer.writerow(row)

            grand_mean_loss.append(np.mean(loss))
            grand_mean_acc.append(np.mean(acc))

        standard_error = np.std(grand_mean_acc) / np.sqrt(len(grand_mean_acc))

        print('[INFO] Standard Error of the mean model after {0} is {1:.4}'.format(n_repeats, standard_error))

    def test_detector(self, test_path, saved_model):

        test_data, test_labels = self.load_dataset(test_path)

        model = load_model(saved_model)

        pred = model.predict(test_data)
        pred = np.argmax(pred, axis=1)

        # y_compare = np.argmax(test_labels, axis=1)
        score = metrics.accuracy_score(test_labels, pred)
        print("Final accuracy: {}".format(score))

        # Compute confusion matrix
        cm = confusion_matrix(test_labels, pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        utils.plot_confusion_matrix(cm, names=['M', 'B'], plot_name='metrics/conf_matrix.png')

        #plot ROC curve
        # pred = [pred[i] for i in np.nonzero(pred)]  # Only positive cases - benign
        utils.plot_roc(pred, test_labels, plot_name='metrics/roc_curve.png')

    def test_detector1(self, image_path, saved_model):
        # load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orig = image.copy()

        # pre-process the image for classification
        image = cv2.resize(image, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network
        print("[INFO] loading network...")
        model = load_model(saved_model)

        # classify the input image
        (malignant, benign) = model.predict(image)[0]

        # build the label
        label = "Benign" if benign > malignant else "Malignant"
        prob = benign if benign > malignant else malignant
        label = "{}: {:.2f}%".format(label, prob * 100)

        # draw the label on the image
        output = imutils.resize(orig, width=200)
        cv2.putText(output, label, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)




if __name__ == '__main__':
    # TODO: data preparation: filters, see papers
    # TODO: random crop: how to send the input to the network?
    # TODO: learned feature visualization in Tensorboard or something else
    # TODO: Histogram tensorboard meaning
    # TODO: cross validation for tuning hyperparmeters skicit-learn

    bcDetector = BreastCancerDetector()
    bcDetector.config(config_filename="configs.txt", config_id=1)  # !!!changed

    dataset = 'E:/work/mias_mammography/DDSM/Train/resized'
    bcDetector.evaluate_models(dataset, n_splits=5, n_repeats=30)

    # now = time.strftime("%w-%m-%Y_%M-%S", time.localtime(time.time()))
    #
    # bcDetector.train(dataset_name=dataset, saved_model='models/model_{}.h5'.format(now))

    # csv_history = utils.load_saved_history('metrics/without_da/plots_run2/training_0-03-2018_32-07.csv')
    # utils.plot_metrics("metrics/without_da/plots_run2/loss_metrics_training_0-03-2018_32-07.png",
    #                    "metrics/without_da/plots_run2/acc_metrics_training_0-03-2018_32-07.png",
    #                    n_epochs=len(csv_history['epoch']),
    #                    history=csv_history)

    # test_img = 'E:/work/mias_mammography/DDSM/Test/resized/benign/Calc_P_00038_CC_1.jpg'
    # test_dataset = 'E:/work/mias_mammography/DDSM/Test/resized'
    # bcDetector.test_detector(test_dataset, 'models/model_1520018177.2960913.h5')
    # bcDetector.test_detector1(test_img, 'models/model_1520018177.2960913.h5')

    # img = 'E:/work/mias_mammography/DDSM/Train/resized/benign/Calc_P_00007_CC_1.jpg'
    # folder = 'E:/work/mias_mammography/DDSM/Train//preview'
    # bcDetector.visualize_data_augmentation(img, folder)