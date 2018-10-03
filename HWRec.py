import keras.optimizers as opt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
#
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
s = int(x_train.shape[0]/2)
x_train1 = x_train[:s]
x_train2 = x_train[s:]
x_train1 /= 255
x_train2 /= 255
x_test /= 255
y_train1 = y_train[:s]
y_train2 = y_train[s:]
y_train1 = np_utils.to_categorical(y_train1)
y_train2 = np_utils.to_categorical(y_train2)
y_test = np_utils.to_categorical(y_test)
#
def base_model() :
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    return model

model = base_model()
optimizer = opt.Adam(lr=0.001, decay=1e-6)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(x_train1, y_train1, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
optimizer = opt.Adam(lr=0.0003, decay=1e-6)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(x_train2, y_train2, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
optimizer = opt.Adam(lr=0.00015, decay=1e-5)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(x_train1, y_train1, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
optimizer = opt.Adam(lr=0.00002, decay=1e-5)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(x_train2, y_train2, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores[1]*100)
model_json = model.to_json()
with open("model4.json", "w") as json_file :
    json_file.write(model_json)
model.save_weights("model4.h5")
# json_file = open("model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")
