from cnn import AnimalClassifier

if __name__ == '__main__':
	classifier = AnimalClassifier()
	classifier.makeModel()
	# classifier.train('data/training', 'data/validation/')
	prediction = classifier.classify('data/validation/airplane/airplane01.tif')
	print('Image is classified as: ', prediction)
