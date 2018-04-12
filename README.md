# BreastCancerTumorClassifier

~~ WORK IN PROGRESS ~~

### Short intro ###
Implementation of a breast cancer tumor classifier (benign or malignant) using deep learning techniques. 

Since I wanted to start as simple as possible, I use the LeNet architecture (with ReLu function, and Adam as optimizer) for classifying the tumors. I made a ToDo list with the things I want to try and implement.

### TO DO List ###
The following steps will be done for three types of image resolutions: 28 x 28, 150 x 150, 256 x 256
* Train a model without data augmentation
* Train a model with simple data augmentation (e.g rotation, shift, vertical flip)
* Train a model using regularization techniques (e.g dropout, L1/L2)
* Train a model with a modified version of LeNet that has a Global Average Pooling layer instead of the fully-connected one.
* Train a model using transfer learning

Another step would be to use a different network or other techniques more specific for medical images.
* UNet
* [Use tissue augmentation](https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12110)

### Dependencies ###
* Python 3.6.3
* Keras 2.1.5
* Tensorflow 1.1.0
* Opencv 3.3.1
* pydicom 0.9.9
* scikit-learn 0.19.1

### Dataset ###
* [CBIS-DDSM (Curated Breast Imaging Subset of DDSM)](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
