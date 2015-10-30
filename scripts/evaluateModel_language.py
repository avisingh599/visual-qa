import sys
import argparse

from keras.models import model_from_json

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import computeLanguageVectorsBatch



def main():
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
	
	print 'Model compiled, weights loaded...'
	labelencoder = joblib.load('../models/labelencoder.pkl')

	nlp = English()
	print 'loaded word2vec features'

	nb_classes = 1000
	

	y_predict_text = []
	for qu in grouper(questions_train, batchSize, fillvalue=questions_train[0]):
		x = computeLanguageVectors(qu,nlp)
		y_predict = model.predict_classes(x, verbose=0)
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

		print len(y_predict_text)

	correct_val=0
	incorrect_val=0
	
	f1 = open(args.results, 'w')
	f1.write('\n'.join(y_predict_text) + '\n')

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

if __name__ == "__main__":
	main()