import sys
from random import shuffle
import argparse

import numpy as np
import scipy.io

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils

from sklearn import preprocessing
from sklearn.externals import joblib

from spacy.en import English

from features import computeVectorsBatch
from utils import grouper, selectFrequentAnswers

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units', type=int, default=1024)
	parser.add_argument('-num_hidden_layers', type=int, default=2)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation', type=str, default='tanh')
	args = parser.parse_args()

	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'
	maxAnswers = 1000
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, maxAnswers)

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,'../models/labelencoder.pkl')
	#y_train = le.transform(answers_train)
	#Y_train = np_utils.to_categorical(y_train, nb_classes)

	#define model, in this case an MLP
	img_dim = 4096
	word_vec_dim = 300
	model = Sequential()
	for i in xrange(args.num_hidden_layers):
		model.add(Dense(args.num_hidden_units, input_dim=img_dim+word_vec_dim, init='uniform'))
		model.add(Activation(args.activation))
		model.add(Dropout(args.dropout))


	model.add(Dense(nb_classes, init='uniform'))
	model.add(Activation('softmax'))
	json_string = model.to_json()
	open('../models/mlp_full_num_hidden_units_1000.json', 'w').write(json_string)

	print 'Compiling model...'
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done...'
	#set up CNN features and word vectors
	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'loaded vgg features'
	image_ids = open('../features/coco/coco_vgg_IDMap.txt').read().splitlines()
	id_map = {}
	for ids in image_ids:
		id_split = ids.split()
		id_map[id_split[0]] = int(id_split[1])

	nlp = English()
	print 'loaded word2vec features...'
	## training
	batchSize = 128
	print 'Training started...'
	numEpochs = 50
	for k in xrange(numEpochs):
		#shuffle the data points before going through them
		index_shuf = range(len(questions_train))
		shuffle(index_shuf)
		questions_train = [questions_train[i] for i in index_shuf]
		answers_train = [answers_train[i] for i in index_shuf]
		images_train = [images_train[i] for i in index_shuf]
		progbar = generic_utils.Progbar(len(questions_train))
		for qu,an,im in zip(grouper(questions_train, batchSize, fillvalue=questions_train[0]), grouper(answers_train, batchSize, fillvalue=answers_train[0]), grouper(images_train, batchSize, fillvalue=images_train[0])):
			X_batch, Y_batch = computeVectorsBatch(qu,an,im,VGGfeatures,nlp,id_map,labelencoder,nb_classes)
			loss = model.train_on_batch(X_batch, Y_batch)
			progbar.add(batchSize, values=[("train loss", loss)])
		#print type(loss)
		if k%10 == 0:
			model.save_weights('../models/mlp_epoch_{:02d}_loss_{:.2f}.hdf5'.format(k,float(loss)) )