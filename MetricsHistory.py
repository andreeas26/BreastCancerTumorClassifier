import numpy as np
import time
from keras.callbacks import Callback
import csv




class MetricsHistory(Callback):

    def __init__(self):
        super().__init__()
        self.mean_loss = 0
        self.mean_bin_acc = 0
        self.mean_val_loss = 0
        self.mean_val_bin_acc = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.bin_acc = []
        self.val_losses = []
        self.val_bin_acc = []

        now = time.strftime("%w-%m-%Y_%M-%S", time.localtime(time.time()))

        header = ['Epoch', 'Batch', 'Loss', 'Binary_acc', 'Val_loss', 'Val_binary_acc']

        csv_file = open('metrics/training_{}.csv'.format(now), 'w')

        self.csv_writer = csv.DictWriter(csv_file, fieldnames=header)
        self.csv_writer.writeheader()

    # def on_train_end(self, logs=None):
    #     self.mean_loss = np.mean(np.array(self.losses, dtype=np.double))
    #     self.mean_bin_acc = np.mean(np.array(self.bin_acc, dtype=np.double))
    #     self.mean_val_loss = np.mean(np.array(self.val_losses, dtype=np.double))
    #     self.mean_val_bin_acc = np.mean(np.array(self.val_bin_acc, dtype=np.double))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.bin_acc.append(logs.get('binary_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_bin_acc.append(logs.get('val_binary_accuracy'))

        row = {'Epoch': self.epoch,
               'Batch': self.batch,
               'Loss': logs.get('loss'),
               'Binary_acc': logs.get('binary_accuracy'),
               'Val_loss': logs.get('val_loss'),
               'Val_binary_acc': logs.get('val_binary_accuracy')}

        self.csv_writer.writerow(row)

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch

    # def on_batch_end(self, batch, logs={}):

        # row = {'Epoch': self.epoch,
        #        'Batch': self.batch,
        #        'Loss': logs.get('loss'),
        #        'Binary_acc': logs.get('binary_accuracy')}
        #
        # self.csv_writer.writerow(row)

    def get_mean_values(self):

        metrics = {'MLoss': self.mean_loss,
               'MBin_acc': self.mean_bin_acc,
               'MVal_loss': self.mean_val_loss,
               'MVal_bin_acc': self.mean_val_bin_acc}

        return metrics