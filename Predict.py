import cv2
from keras.models import model_from_json
import sys
try:
	imgname = sys.argv[1]
except:
	print("No image provided")
	sys.exit()
img = cv2.imread('digits/'+imgname, cv2.IMREAD_GRAYSCALE)
resized = cv2.resize(img, (28, 28))
cv2.imshow("resized", resized)
cv2.waitKey(0)
resized = resized.reshape(1, 1, 28, 28).astype('float32')
resized = resized / 255
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
'''with open('ComplexCNN.pkl', 'rb') as f:
	model = pickle.load(f)
	f.close()'''
pred = model.predict(resized)
i = 0
for cp in list(map(lambda x: str(float(x)*100), list(pred)[0])):
	print(i, '-->', cp)
	i += 1
print(pred.argmax(axis=-1))

'''from keras.utils import np_utils
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
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
with open('ComplexCNN.pkl', 'wb') as f:
	pickle.dump(model, f)
	f.close()
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))'''