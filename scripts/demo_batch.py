import argparse
import random
from PIL import Image
import subprocess
from os import listdir
from os.path import isfile, join

from keras.models import model_from_json

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import get_questions_tensor_timeseries, get_images_matrix, get_answers_matrix

def main():
	'''
	Before runnning this demo ensure that you have some images from the MS COCO validation set
	saved somewhere, and update the image_dir variable accordingly
	Also, this demo is designed to run with the models released with the visual-qa repo, if you
	would like to get use it with some other model (say an MLP based model or a langauge-only model)
	you will have to make some changes.
	'''
	image_dir = '../../vqa_images/'
	local_images = [ f for f in listdir(image_dir) if isfile(join(image_dir,f)) ]	
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, default='../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1.json')
	parser.add_argument('-weights', type=str, default='../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_num_hidden_layers_lstm_1_epoch_199.hdf5')
	parser.add_argument('-sample_size', type=int, default=25)
	args = parser.parse_args()
	
	model = model_from_json(open(args.model).read())
	model.load_weights(args.weights)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Model loaded and compiled'
	images_val = open('../data/preprocessed/images_val2014.txt', 
						'r').read().decode('utf8').splitlines()

	nlp = English()
	print 'Loaded word2vec features'
	labelencoder = joblib.load('../models/labelencoder.pkl')

	vgg_model_path = '../features/coco/vgg_feats.mat'
	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'Loaded vgg features'
	image_ids = open('../features/coco_vgg_IDMap.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	image_sample = random.sample(local_images, args.sample_size)

	for image in image_sample:
		p = subprocess.Popen(["display", image_dir + image])
		q = unicode(raw_input("Ask a question about the image:"))	
		coco_id = str(int(image[-16:-4]))
		timesteps = len(nlp(q)) #questions sorted in descending order of length
		X_q = get_questions_tensor_timeseries([q], nlp, timesteps)
		X_i = get_images_matrix([coco_id], img_map, VGGfeatures)
		X = [X_q, X_i]
		y_predict = model.predict_classes(X, verbose=0)
		print labelencoder.inverse_transform(y_predict)
		raw_input('Press enter to continue...')
		p.kill()

if __name__ == "__main__":
	main()
