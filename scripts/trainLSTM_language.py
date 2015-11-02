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

from features import get_questions_tensor_timeseries, get_answers_matrix
from utils import grouper, selectFrequentAnswers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units', type=int, default=512)
	parser.add_argument('-num_lstm_layers', type=int, default=2)
	parser.add_argument('-dropout', type=float, default=0.2)
	parser.add_argument('-activation', type=str, default='tanh')
	args = parser.parse_args()

	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	questions_lengths_train = open('../data/preprocessed/questions_lengths_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	max_answers = 1000
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, max_answers)

	print 'Loaded questions, sorting by length...'
	questions_lengths_train, questions_train, answers_train = (list(t) for t in zip(*sorted(zip(questions_lengths_train, questions_train, answers_train))))
	
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
	model_file_name = '../models/lstm_language_only_num_hidden_units_' + str(args.num_hidden_units) + '_num_lstm_layers_' + str(args.num_lstm_layers) + '_dropout_' + str(args.dropout)
	open(model_file_name  + '.json', 'w').write(json_string)
	
	print 'Compiling model...'
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done...'

	#set up word vectors
	nlp = English()
	print 'loaded word2vec features...'

	## training
	print 'Training started...'
	numEpochs = 100
	model_save_interval = 5
	batchSize = 128
	for k in xrange(numEpochs):

		progbar = generic_utils.Progbar(len(questions_train))

		for qu_batch,an_batch,im_batch in zip(grouper(questions_train, batchSize, fillvalue=questions_train[0]), 
												grouper(answers_train, batchSize, fillvalue=answers_train[0]), 
												grouper(images_train, batchSize, fillvalue=images_train[0])):
			timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
			X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
			Y_batch = get_answers_matrix(an_batch, labelencoder)
			loss = model.train_on_batch(X_q_batch, Y_batch)
			progbar.add(batchSize, values=[("train loss", loss)])

		
		if k%model_save_interval == 0:
			model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

	model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k+1))

if __name__ == "__main__":
	main()
