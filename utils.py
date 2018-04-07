import matplotlib.pylab as pylab
import matplotlib.cm as cm
import numpy as np
import cv2
import dicom
import csv
import os
from sklearn.metrics import roc_curve, auc
from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues, plot_name='confusion_matrix.png'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(plot_name)


# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred, y, plot_name='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(plot_name)


def plot_metrics(loss_plot_name, acc_plot_name, n_epochs, history):
    # plot the training loss and accuracy
    print('[INFO] Plotting the metrics...')
    plt.style.use("ggplot")
    plt.figure(1)

    plt.plot(np.arange(0, n_epochs), history["loss"], marker='o', label="train_loss")
    plt.plot(np.arange(0, n_epochs), history["val_loss"], marker='o', label="val_loss")
    plt.title("Training and Validation Loss on Breast Cancer detector")
    plt.axes().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(loss_plot_name)

    # clear the figure
    plt.gcf().clear()

    plt.figure(2)
    plt.plot(np.arange(0, n_epochs), history["binary_accuracy"], marker='o', label="train_acc")
    plt.plot(np.arange(0, n_epochs), history["val_binary_accuracy"], marker='o', label="val_acc")
    plt.title("Training and Validation Accuracy on Breast Cancer detector")
    plt.axes().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(acc_plot_name)

    # clear the figure
    plt.gcf().clear()

def load_saved_history(csv_history_filename):
    csv_history = {'epoch': [],
                   'loss': [],
                   'binary_accuracy': [],
                   'val_loss': [],
                   'val_binary_accuracy': []}

    csv_hist = open(csv_history_filename, 'r')
    # csv_header = ['Epoch', 'Batch', 'Loss', 'Binary_acc', 'Val_loss', 'Val_binary_acc']
    csv_reader = csv.DictReader(csv_hist)

    for r in csv_reader:
        csv_history['epoch'].append(int(r['Epoch']))
        csv_history['loss'].append(float(r['Loss']))
        csv_history['binary_accuracy'].append(float(r['Binary_acc']))
        csv_history['val_loss'].append(float(r['Val_loss']))
        csv_history['val_binary_accuracy'].append(float(r['Val_binary_acc']))

    return csv_history


def dicom2jpg(dcm_filename, jpg_filename, new_size = None):
    ds = dicom.read_file(dcm_filename)
    jpg = ds.pixel_array

    # if new_size is not None:
    #     jpg = cv2.resize(jpg, new_size, interpolation=cv2.INTER_AREA)
    pylab.imsave(jpg_filename, jpg, cmap=cm.Greys_r)


def parse_ddsm_csv(csv_filename):
    patients = []
    img_views = []
    abnormality = []
    pathology = []
    dcm_paths = []

    with open(csv_filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for r in reader:
            patients.append(r['patient_id'])
            img_views.append(r['image view'])
            abnormality.append(r['abnormality id'])
            pathology.append(r['pathology'])
            dcm_paths.append(r['image file path'])

    return patients, img_views, abnormality, pathology, dcm_paths


def convert_ddsm2jpgs(csv_filename, in_folder, out_folder):
    patients, img_views, abnormality, pathology, dcm_paths = parse_ddsm_csv(csv_filename)

    print('Processing DDSM dataset {}...'.format(csv_filename))

    for i in range(0, len(dcm_paths)):
        print(i)

        dcm_filename = in_folder + '/' + dcm_paths[i]

        if pathology[i] == 'MALIGNANT':
            img_filename = '{0}/malignant/{1}_{2}_{3}.jpg'.format(out_folder, patients[i], img_views[i], abnormality[i])
        else:
            img_filename = '{0}/benign/{1}_{2}_{3}.jpg'.format(out_folder, patients[i], img_views[i], abnormality[i])

        dicom2jpg(dcm_filename, img_filename)


def resize_jpgs(in_folder, out_folder, new_size):
    in_paths = glob('{}/*.jpg'.format(in_folder))

    print('Resizing {} images to {}...'.format(len(in_paths), new_size))
    i = 1
    for p in in_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, new_size, cv2.INTER_AREA)
        filename = os.path.basename(p)
        cv2.imwrite('{}/{}'.format(out_folder, filename), img)
        print(i)
        i += 1

# dicom2jpg("test.dcm", "test.jpg")
# parse_ddsm_csv('E:/work/mias_mammography/DDSM/mass_case_description_train_set.csv')
# c = 'E:/work/mias_mammography/DDSM/mass_case_description_test_set.csv'
# c = 'E:/work/mias_mammography/DDSM/calc_case_description_test_set.csv'
# in_folder = 'E:/work/mias_mammography/DDSM/Train/Calc/malignant'
# out_folder = 'E:/work/mias_mammography/DDSM/Train/Calc/resized/malignant'

# resize_jpgs(in_folder, out_folder, (28, 28))
# convert_ddsm2jpgs(c, in_folder, out_folder)