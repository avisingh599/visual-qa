import numpy as np
import scipy.io
from random import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, RepeatVector
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor

from sklearn.externals import joblib
from sklearn import preprocessing

from spacy.en import English

from features import computeVectorsBatchTimeSeries
from utils import grouper, selectFrequentAnswers

if __name__ == "__main__":

	featureDim= 300
	maxLen = 20
	nb_classes = 1000

	#get the data
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
	#defining our LSTM based model
	numHiddenUnits = 512
	image_model = Sequential()
	image_model.add(Dense(featureDim, input_dim=4096, init='uniform', activation='linear'))
	image_model.add(RepeatVector(1))
	image_model.add(LSTM(output_dim = numHiddenUnits, return_sequences=True, input_shape=(1, featureDim)))
	#print image_model.output_shape
	#512 hidden units in LSTM layer. 300-dimnensional word vectors.
	language_model = Sequential()
	language_model.add(LSTM(output_dim = numHiddenUnits, return_sequences=True, input_shape=(maxLen, featureDim)))
	#print language_model.output_shape

	model = Sequential()
	model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
	#print model.output_shape
	model.add(LSTM(numHiddenUnits, return_sequences=False))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	json_string = model.to_json()
	open('../models/lstm_2_numHiddenUnits_512.json', 'w').write(json_string)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done'

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'loaded vgg features'
	image_ids = open('../features/coco/coco_vgg_IDMap.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	nlp = English()
	print 'loaded word2vec features...'
	## training
	batchSize = 128
	print 'Training started...'
	numEpochs = 10
	for k in xrange(numEpochs):
		#shuffle the data points before going through them
		index_shuf = range(len(questions_train))
		shuffle(index_shuf)
		questions_train = [questions_train[i] for i in index_shuf]
		answers_train = [answers_train[i] for i in index_shuf]
		images_train = [images_train[i] for i in index_shuf]

		progbar = generic_utils.Progbar(len(questions_train))
		for qu,an,im in zip(grouper(questions_train, batchSize, fillvalue=questions_train[0]), grouper(answers_train, batchSize, fillvalue=answers_train[0]), grouper(images_train, batchSize, fillvalue=images_train[0])):
			X_i_batch, X_q_batch, Y_batch = computeVectorsBatchTimeSeries(qu,an,im,VGGfeatures,img_map,nlp,labelencoder,nb_classes, maxLen)
			loss = model.train_on_batch([X_i_batch,X_q_batch], Y_batch)
			progbar.add(batchSize, values=[("train loss", loss)])
		model.save_weights('../models/lstm_2_epoch_{:02d}_loss_{:.2f}.hdf5'.format(k,float(loss)))
