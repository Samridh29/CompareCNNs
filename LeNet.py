from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K


class LeNet:
    @staticmethod
    def build_lenet(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
    
        model.add(Conv2D(6, (5*5), padding = "same", input_shape = inputShape))
        model.add(Activation("Relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
        model.add(Conv2D(6, (5*5), padding = "same"))
        model.add(Activation("Relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
        model.add(Conv2D(120, (5,5), padding = "same"))
        model.add(Flatten())
        model.add(Dense(84))
        model.add(Activation("Relu"))
        model.add(Dense(classes))
        model.add(Activation("Softmax"))

        return model

