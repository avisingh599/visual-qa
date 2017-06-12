import sys
from random import shuffle
import argparse

import numpy as np
import spacy

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
	parser.add_argument('-num_epochs', type=int, default=100)
	parser.add_argument('-model_save_interval', type=int, default=5)
	parser.add_argument('-batch_size', type=int, default=128)
	parser.add_argument('-word_vector', type=str, default='')
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
        # Code to choose the word vectors, default is Goldberg but GLOVE is preferred
        if args.word_vector == 'glove':
            nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
        else:
            nlp = English()

	print 'loaded ' + args.word_vector + ' word2vec features...'

	## training
        # Moved few variables to args.parser (num_epochs, batch_size, model_save_interval)
	print 'Training started...'
	for k in xrange(args.num_epochs):

		progbar = generic_utils.Progbar(len(questions_train))

		for qu_batch,an_batch,im_batch in zip(grouper(questions_train, args.batch_size, fillvalue=questions_train[0]), 
												grouper(answers_train, args.batch_size, fillvalue=answers_train[0]), 
												grouper(images_train, args.batch_size, fillvalue=images_train[0])):
			timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
			X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
			Y_batch = get_answers_matrix(an_batch, labelencoder)
			loss = model.train_on_batch(X_q_batch, Y_batch)
			# fix for the Keras v0.3 issue #9
			progbar.add(args.batch_size, values=[("train loss", loss[0])])

		
		if k%args.model_save_interval == 0:
			model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

	model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k+1))

if __name__ == "__main__":
	main()
