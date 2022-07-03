from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K

class VGG:
    @staticmethod
    def build_vgg(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)
        chandim = -1

        if K.image.data_format() == "channels_first":
            inputShape = (depth, height, width)
            chandim = 1

        model.add(Conv2D(64, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(BatchNormalization(chandim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)), activation = "relu")
        model.dropout(0.25)

        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(BatchNormalization(chandim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)), activation = "relu")
        model.dropout(0.25)

        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(BatchNormalization(chandim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)), activation = "relu")
        model.dropout(0.25)

        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(BatchNormalization(chandim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)), activation = "relu")
        model.dropout(0.25)

        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(BatchNormalization(chandim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2)), activation = "relu")
        model.dropout(0.25)

        model.add(Flatten())
        model.add(Dense(25088))
        model.add(Activation("relu"))
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

        

