import numpy as np
from keras import backend as K

class CustomAugmentation(object):
    def __init__(self,
                 random_crop_size=None):
        self.random_crop_size = random_crop_size

    def __call__(self, image):

        if K.image_data_format() == "channels_first":
            height, width = image.shape[1:]
        else:
            height, width = image.shape[:2]

        dy, dx = self.random_crop_size

        if width < dx or height < dy:
            return None

        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)

        return image[:, y:(y + dy), x:(x + dx)]
