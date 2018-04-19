from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os.path


class AnimalClassifier:

	def makeModel(self):
		print('Initializing the convolutional neural network..')
		self.classifier = Sequential()

		#conv1
		self.classifier.add(Conv2D(32, (3,3), input_shape=(256,256,3), activation='relu'))
		#pool1
		self.classifier.add(MaxPooling2D(pool_size = (2,2)))
		#conv2
		self.classifier.add(Conv2D(32,(3,3), activation='relu'))
		#pool2
		self.classifier.add(MaxPooling2D(pool_size= (2,2)))
		#conv3
		self.classifier.add(Conv2D(64,(3,3), activation='relu'))
		#pool3
		self.classifier.add(MaxPooling2D(pool_size= (2,2)))

		#flatten
		self.classifier.add(Flatten())

		#fully connected
		self.classifier.add(Dense(units=64, activation='relu'))
		#self.classifier.add(Dropout(0.5))
		self.classifier.add(Dense(units=10, activation='softmax'))
		#compiling the conv neural network
		self.classifier.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
		print('Done Compiling the network..')

	def train(self, train_path, test_path):
	
		print('Starting to train the dataset', train_path)
		
		# training data		
		train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
		train_set = train_datagen.flow_from_directory(train_path, target_size=(256,256), batch_size=32, class_mode='categorical')
		print(train_set.classes)
		
		#testing data		
		test_datagen = ImageDataGenerator(rescale = 1./255)
		test_set = test_datagen.flow_from_directory(test_path, target_size= (256,256), batch_size=32, class_mode = 'categorical')
		self.classifier.fit_generator(train_set, steps_per_epoch= 288, epochs=5, validation_data = test_set, validation_steps = 32)
		
		#saving the model 
		print('saving the trained model')
		self.classifier.save('my_model_1.h5')
		print('Done training the dataset')

	def classify(self, path):
	
		print('classifying ',path)
		test_image = image.load_img(path, target_size=(256,256))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis=0)
		
		if(hasattr(self, 'classfier')==False):
			if(os.path.exists('my_model_1.h5')):
				self.classifier = load_model('my_model_1.h5')
			else:
				print('model is not trained')
		
		result = self.classifier.predict(test_image)
		prediction = result
		return prediction


