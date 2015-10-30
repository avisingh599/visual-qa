import sys
import argparse

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, RepeatVector
from keras.layers.recurrent import LSTM

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import computeVectors, computeVectorsTimeSeries


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	#parser.add_argument('-model', type=str, required=True)
	parser.add_argument('-weights', type=str, required=True)
	parser.add_argument('-results', type=str, required=True)
	args = parser.parse_args()

	featureDim = 300
	numHiddenUnits = 512
	maxLen = 20
	nb_classes = 1000

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

	#model = model_from_json(open(args.model).read())
	model.load_weights(args.weights)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	questions_val = open('../data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
	answers_val = open('../data/preprocessed/answers_val2014.txt', 'r').read().decode('utf8').splitlines()
	images_val = open('../data/preprocessed/images_val2014.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'
	
	print 'Model compiled, weights loaded...'
	labelencoder = joblib.load('../models/labelencoder.pkl')

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'loaded vgg features'
	image_ids = open('../features/coco/coco_vgg_IDMap.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	nlp = English()
	print 'loaded word2vec features'

	nb_classes = 1000
	y_predict_text = []
	for q,an,im,i in zip(questions_val,answers_val,images_val,xrange(len(questions_val))):
		x_i,x_q = computeVectorsTimeSeries(q,an,im,VGGfeatures, nlp, img_map, labelencoder, nb_classes, maxLen)
		y_predict = model.predict_classes([x_i,x_q], verbose=0)
		y_predict_text.append(labelencoder.inverse_transform(y_predict))
		#print y_predict_text[-1]
		if i%100 == 0:
			print i

	correct_val=0
	incorrect_val=0
	f1 = open(args.results, 'w')

	for prediction,truth in zip(y_predict_text, answers_val):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1

		if temp_count>2:
			correct_val+=1
		else:
			incorrect_val+=1
		#print type(prediction)
		f1.write(prediction.encode("utf-8"))
		f1.write('\n')

	f1.write(str(float(correct_val)/(incorrect_val+correct_val)))
	print float(correct_val)/(incorrect_val+correct_val)




