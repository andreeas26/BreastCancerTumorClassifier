# import necessary packages
from keras.models import Sequential
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# The LeNet architecture with a Global Average Pooling Layer
# instead of the last fully-connected layer


class LeNetGAPL:
    @staticmethod
    def build(width, height, depth, classes):
        # initiliaze the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using " channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layes
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => Global Average pooling
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed model
        return model