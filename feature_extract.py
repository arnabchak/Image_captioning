# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 

@author: arnab
"""



from os import listdir
from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	model = InceptionV3(weights='imagenet')
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print(model.summary())
	features = dict()
	for name in listdir(directory):
		filename = directory + '/' + name
		image = load_img(filename, target_size=(299, 299))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		image_id = name.split('.')[0]
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = 'Pineapple_test'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('features_pineapple_test.pkl', 'wb'))

