import tensorflow
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM



import random

model_arch = {
    'LSTM': [100, 1, 36],
    'Dense': [72, 'relu'],
    'Dense1': [50, 'relu'],
    'Dense2': [36, 'relu'],
    'Dense3': [28, 'relu'],
    'Dense4': [10, 'softmax']
}



# Create an initial LSTM Model
def create_seed_model(trainedLayers=0):

    lay_count = 0

    if trainedLayers > 0:
        randomlist = random.sample(range(0, len(model_arch)), trainedLayers)

        print(" ---------------Trained layers------- ", flush=True)
        print(randomlist, flush=True)
        print(" --------------------------------------- ", flush=True)


        with open('results/layers.txt', '+a') as f:
            print(randomlist, file=f)

        model = Sequential()
        for key, item in model_arch.items():
            if lay_count in randomlist:
                if key in 'LSTM':
                    model.add(LSTM(item[0], input_shape=(item[1], item[2]), trainable=True))
                else:
                    model.add(tensorflow.keras.layers.Dense(item[0], activation=item[1], trainable=True))
            else:
                if key in 'LSTM':
                    model.add(LSTM(item[0], input_shape=(item[1], item[2]), trainable=False))
                else:
                    model.add(tensorflow.keras.layers.Dense(item[0], activation=item[1], trainable=False))

            lay_count += 1

        # model.add(Dense(10, activation='softmax'))
        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

        print(" --------------------------------------- ", flush=True)
        print(" ------------------MODEL CREATED------------------ ", flush=True)
        print(" --------------------------------------- ", flush=True)

    else:
####### Original model
        model = Sequential()
        model.add(LSTM(100, input_shape=(1, 36)))
        model.add(tensorflow.keras.layers.Dense(72, activation='relu'))
        model.add(tensorflow.keras.layers.Dense(50, activation='relu'))
        model.add(tensorflow.keras.layers.Dense(36, activation='relu'))
        model.add(tensorflow.keras.layers.Dense(28, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=['accuracy'])
    return model
