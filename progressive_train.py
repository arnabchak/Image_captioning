
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 1 

@author: arnab
"""


from numpy import array,asarray, zeros
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU, BatchNormalization
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2


def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
 
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# loading clean descriptions
def load_clean_descriptions(filename, dataset):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			descriptions[image_id].append(desc)
	return descriptions
 

def load_photo_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features
 
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	for desc in desc_list:
		seq = tokenizer.texts_to_sequences([desc])[0]
		for i in range(1, len(seq)):
			in_seq, out_seq = seq[:i], seq[i]
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)


"""def load_glove_embedding(vocab_size,tokenizer):
    embeddings_index = dict()
    f = open('glove.6B.300d.txt',encoding="utf8")
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    #print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix   """     
 
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dense(256)(inputs1)
    #fe2 = BatchNormalization(momentum=0.9)(fe1)
    fe3 = Dropout(0.5)(fe1)
    fe4 = LeakyReLU(alpha=0.1)(fe3)
    
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size,256,mask_zero=True)(inputs2)
    #se2 =  BatchNormalization(momentum=0.9)(se1)
    se3 = Dropout(0.5)(se1)
    se4 = LSTM(256,return_sequences=True)(se3)
    se6 = LeakyReLU(alpha=0.1)(se4)
    # decoder model
    captioner1 = add([fe4, se6])
    captioner2 = Dense(256)(captioner1)
    #captioner3 = BatchNormalization(momentum=0.9)(captioner2)
    captioner4 = LeakyReLU(alpha=0.1)(captioner2)
    outputs = Dense(vocab_size, activation='softmax')(captioner4)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #model.load_weights("incv3_model-ep001-loss3.379-val_loss3.833.h5")
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()
    #plot_model(model, to_file='model_incv3.png', show_shapes=True)
    return model
 
# data generator
def data_generator(descriptions, photos, tokenizer, max_length):
	while 1:
		for key, desc_list in descriptions.items():
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

# load training dataset
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# load test dataset
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))

test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))


#embed_matrix = load_glove_embedding(vocab_size,tokenizer) 
# define the model
model = define_model(vocab_size, max_length)
# train the model
epochs = 40
steps = len(train_descriptions)
for i in range(epochs):
      filepath = 'BI-incv3-MODEL-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
      checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
      generator1 = data_generator(train_descriptions, train_features, tokenizer, max_length)
      generator2 = data_generator(test_descriptions, test_features, tokenizer, max_length) 
      model.fit_generator(generator1, epochs=1, steps_per_epoch=steps, verbose=1,callbacks=[checkpoint],validation_data=generator2, validation_steps=len(test_descriptions))

