import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...\n")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#Scaling the images
x_train = x_train / 255.0
x_test = x_test / 255.0

#Reshaping the images
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

#Initialization of the model
print("\nModel Initialization\n")
model = Sequential()

#Adding the input layer
model.add(Dense(512, activation='relu', input_shape=(784,)))

#Adding 4 hidden layers
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

#Adding the output layer
model.add(Dense(10, activation='softmax'))

#Compiling 
print("\nmodel compilation\n")
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Training the model
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

#Evaluating the model
score = model.evaluate(x_test, y_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

#Accuracy History
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Loss History
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()