import sys
from random import shuffle
import argparse

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, generic_utils

from sklearn import preprocessing
from sklearn.externals import joblib

from spacy.en import English

from features import computeLanguageVectorsBatch
from utils import grouper, selectFrequentAnswers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units', type=int, default=1024)
	parser.add_argument('-num_hidden_layers', type=int, default=2)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation', type=str, default='tanh')
	args = parser.parse_args()

	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	maxAnswers = 1000
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, maxAnswers)

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,'../models/labelencoder.pkl')

	word_vec_dim = 300
	model = Sequential()
	for i in xrange(args.num_hidden_layers):
		model.add(Dense(args.num_hidden_units, input_dim=word_vec_dim, init='uniform'))
		model.add(Activation(args.activation))
		model.add(Dropout(args.dropout))

	model.add(Dense(nb_classes, init='uniform'))
	model.add(Activation('softmax'))

	json_string = model.to_json()
	model_file_name = '../models/mlp_language_num_hidden_units_' + str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)
	open(model_file_name  + '.json', 'w').write(json_string)
	
	print 'Compiling model...'
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done...'

	#set up word vectors
	nlp = English()
	print 'loaded word2vec features...'
	## training
	batchSize = 128
	print 'Training started...'
	numEpochs = 100
	for k in xrange(numEpochs):
		#shuffle the data points before going through them
		index_shuf = range(len(questions_train))
		shuffle(index_shuf)
		questions_train = [questions_train[i] for i in index_shuf]
		answers_train = [answers_train[i] for i in index_shuf]
		progbar = generic_utils.Progbar(len(questions_train))
		for qu,an in zip(grouper(questions_train, batchSize, fillvalue=questions_train[0]), grouper(answers_train, batchSize, fillvalue=answers_train[0])):
			X_batch, Y_batch = computeLanguageVectorsBatch(qu,an,nlp,labelencoder,nb_classes)
			loss = model.train_on_batch(X_batch, Y_batch)
			progbar.add(batchSize, values=[("train loss", loss)])
		#print type(loss)
		if k%10 == 0:
			model.save_weights(model_file_name + '_epoch_{:02d}_loss_{:.2f}.hdf5'.format(k,float(loss)))

	model.save_weights(model_file_name + '_epoch_{:02d}_loss_{:.2f}.hdf5'.format(k+1,float(loss)))

if __name__ == "__main__":
	main()
