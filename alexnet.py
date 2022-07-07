from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

class AlexNet:
    @staticmethod
    def build_alexnet(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        model.add(Conv2D(96, (11,11), strides = (4,4), input_shape = inputShape, padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

        model.add(Conv2D(256, (5,5), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

        model.add(Conv2D(384, (3,3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(384, (3,3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3,3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("Softmax"))

        return model
