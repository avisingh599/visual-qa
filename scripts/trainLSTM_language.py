import sys
from random import shuffle
import argparse

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils

from sklearn import preprocessing
from sklearn.externals import joblib

from spacy.en import English

from features_2 import computeLanguageVectorsTimeSeries
from utils import grouper, selectFrequentAnswers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units', type=int, default=512)
	parser.add_argument('-num_lstm_layers', type=int, default=2)
	parser.add_argument('-dropout', type=float, default=0.2)
	parser.add_argument('-activation', type=str, default='tanh')
	args = parser.parse_args()

	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	max_answers = 1000
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, max_answers)

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,'../models/labelencoder.pkl')
	max_len = 30 #25 is max for training, 27 is max for validation
	word_vec_dim = 300

	model = Sequential()
	model.add(LSTM(output_dim = args.num_hidden_units, activation='tanh', 
			return_sequences=True, input_shape=(max_len, word_vec_dim)))
	model.add(Dropout(args.dropout))
	model.add(LSTM(args.num_hidden_units, return_sequences=False))
	model.add(Dense(nb_classes, init='uniform'))
	model.add(Activation('softmax'))

	json_string = model.to_json()
	model_file_name = '../models/lstm_language_num_hidden_units_' + str(args.num_hidden_units) + '_num_lstm_layers_' + str(args.num_lstm_layers) + '_dropout_' + str(args.dropout)
	open(model_file_name  + '.json', 'w').write(json_string)
	
	print 'Compiling model...'
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done...'

	#set up word vectors
	nlp = English()
	print 'loaded word2vec features...'
	## training
	print 'Training started...'
	numEpochs = 50
	for k in xrange(numEpochs):
		#shuffle the data points before going through them
		index_shuf = range(len(questions_train))
		shuffle(index_shuf)
		questions_train = [questions_train[i] for i in index_shuf]
		answers_train = [answers_train[i] for i in index_shuf]
		progbar = generic_utils.Progbar(len(questions_train))
		for qu,an in zip(questions_train,answers_train):
			X_batch, Y_batch = computeLanguageVectorsTimeSeries(qu,an,nlp,labelencoder,nb_classes, max_len)
			loss = model.train_on_batch(X_batch, Y_batch)
			progbar.add(1, values=[("train loss", loss)])
		#print type(loss)
		if k%10 == 0:
			model.save_weights(model_file_name + '_epoch_{:02d}_loss_{:.2f}.hdf5'.format(k,float(loss)))

	model.save_weights(model_file_name + '_epoch_{:02d}_loss_{:.2f}.hdf5'.format(k+1,float(loss)))

if __name__ == "__main__":
	main()
