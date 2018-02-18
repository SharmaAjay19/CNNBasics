import cv2

img = cv2.imread('digits/demo.png', cv2.IMREAD_GRAYSCALE)
resized = cv2.resize(img, (28, 28))
#cv2.imshow("resized", resized)
#cv2.waitKey(0)
resized = resized.reshape(1, 1, 28, 28).astype('float32')
resized = resized / 255
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
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#Convert 784 vector into 1x28x28 shape for Convolutional Neural Network
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_resized = numpy.array([4])
y_resized = np_utils.to_categorical(y_resized)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=500)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


score = model.evaluate(resized, y_resized)
print("CNN Error: %.2f%%" % (100-scores[1]*100))