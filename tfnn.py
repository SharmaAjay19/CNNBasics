import pickle
# Plot ad hoc mnist instances
load = True
if load:
	from keras.datasets import mnist
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	pickle.dump(X_train, open('x_train.pkl', 'wb'))
	pickle.dump(y_train, open('y_train.pkl', 'wb'))
	pickle.dump(X_test, open('x_test.pkl', 'wb'))
	pickle.dump(y_test, open('y_test.pkl', 'wb'))
else:
	X_train = pickle.load(open('x_train.pkl', 'rb'))
	y_train = pickle.load(open('y_train.pkl', 'rb'))
	X_test  = pickle.load(open('x_test.pkl', 'rb'))
	y_test  = pickle.load(open('y_test.pkl', 'rb'))
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
# plot 4 images as gray scale
'''plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()'''
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))