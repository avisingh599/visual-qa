import sys
import argparse

from keras.models import model_from_json

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import computeVectors, computeVectorsTimeSeries


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, required=True)
	parser.add_argument('-weights', type=str, required=True)
	parser.add_argument('-results', type=str, required=True)
	args = parser.parse_args()

	model = model_from_json(open(args.model).read())
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
	f1 = open(args.results, 'w')

	y_predict_text = []
	for q,an,im,i in zip(questions_val,answers_val,images_val,xrange(len(questions_val))):
		x = computeVectors(q,an,im,VGGfeatures, nlp, img_map, labelencoder, nb_classes)
		y_predict = model.predict_classes(x, verbose=0)
		y_predict_text.append(labelencoder.inverse_transform(y_predict[-1]))

		f1.write(y_predict_text[-1])
		f1.write('\n')

		if i%100 == 0:
			print i

	correct_val=0
	incorrect_val=0
	

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

	f1.write(str(float(correct_val)/(incorrect_val+correct_val)))
	f1.close()
	print float(correct_val)/(incorrect_val+correct_val)




