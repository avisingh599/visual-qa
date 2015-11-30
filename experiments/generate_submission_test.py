import json
import sys
import argparse
from progressbar import Bar, ETA, Percentage, ProgressBar    
from keras.models import model_from_json

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

sys.path.insert(0, '../scripts/')
from features import get_questions_tensor_timeseries, get_images_matrix, get_answers_matrix
from utils import grouper

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, required=True)
	parser.add_argument('-weights', type=str, required=True)
	parser.add_argument('-results', type=str, required=True)
	args = parser.parse_args()

	model = model_from_json(open(args.model).read())
	model.load_weights(args.weights)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	questions_test = open('../data/preprocessed/questions_test-dev2015.txt', 
						'r').read().decode('utf8').splitlines()
	questions_lengths_test = open('../data/preprocessed/questions_lengths_test-dev2015.txt', 
								'r').read().decode('utf8').splitlines()
	questions_id_test = open('../data/preprocessed/questions_id_test-dev2015.txt', 
								'r').read().decode('utf8').splitlines()
	images_test = open('../data/preprocessed/images_test-dev2015.txt', 
						'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats_test.mat'
	
	questions_lengths_test, questions_test, images_test, questions_id_test = (list(t) for t in zip(*sorted(zip(questions_lengths_test, questions_test, images_test, questions_id_test))))

	print 'Model compiled, weights loaded'
	labelencoder = joblib.load('../models/labelencoder_trainval.pkl')

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'Loaded vgg features'
	image_ids = open('../features/coco_vgg_IDMap_test.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	nlp = English()
	print 'Loaded word2vec features'

	nb_classes = 1000
	y_predict_text = []
	batchSize = 128
	widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
           ' ', ETA()]
	pbar = ProgressBar(widgets=widgets)

	for qu_batch,im_batch in pbar(zip(grouper(questions_test, batchSize, fillvalue=questions_test[-1]), 
												grouper(images_test, batchSize, fillvalue=images_test[-1]))):
		timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
		X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
		if 'language_only' in args.model:
			X_batch = X_q_batch
		else:
			X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
			X_batch = [X_q_batch, X_i_batch]
		y_predict = model.predict_classes(X_batch, verbose=0)
		y_predict_text.extend(labelencoder.inverse_transform(y_predict))

	results = []
	
	f1 = open(args.results, 'w')
	for prediction, question, question_id, image in zip(y_predict_text, questions_test, questions_id_test, images_test):
		answer = {}
		answer['question_id'] = int(question_id)
		answer['answer'] = prediction
		results.append(answer)

		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(image.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(question_id.encode('utf-8'))
		f1.write('\n')
		f1.write('\n')

	f1.close()

	f2 = open('../results/submission_test-dev2015.json', 'w')
	f2.write(json.dumps(results))
	f2.close()
	print 'Results saved to', args.results

if __name__ == "__main__":
	main()