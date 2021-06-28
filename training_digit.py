
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

# data, split between train and test sets
#load_data() method return us the training data, its labels and also the testing data and its labels
(trainX, trainY), (testX, testY) = mnist.load_data()

print(trainX.shape, trainY.shape)

#Perform some operations and proced the data to make it ready fo our cnn
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#convert class vectors to binary class matrices
trainY = keras.utils.to_categorical(trainY, num_classes= 10)
testY = keras.utils.to_categorical(testY, num_classes= 10)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255
print('TrainX shape: ', trainX.shape)
print(trainX.shape[0],'train sample')
print(testX.shape[0], 'test sample')

#create cnn Model
BS = 128
num_classes = 10
EPOCHS = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape= input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Model.fit() function of keras will start the training of the model. it takes the training data, validtion data, epochs, and batch_size
hist = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1, validation_data=(testX, testY))
print('The model has successfully trained!!')

#save model in file h5
model.save('mnist.h5')
print('Saving the model as mnist.h5. Done!')

#make image test train, val
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), hist.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, N), hist.history["val_accuracy"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test Accuracy',score[1])