import numpy as np
from keras.utils import np_utils


def question2VecTimeSeries(nlp, q, feature_dim, max_len):
	doc = nlp(q)
	if len(doc) > max_len:
		outputvec = np.zeros((max_len, feature_dim))

	else:
		outputvec = np.zeros((len(doc), feature_dim))
	
	for i in xrange(len(doc)):
		if i<max_len:
			outputvec[i, :] = doc[i].vector

	return outputvec

def question2VecSum(nlp, q, feature_dim):
	outputvec = np.zeros((feature_dim,))
	try:
		doc = nlp(q)
	except:
		print 'error parsing in spacy:'
		print q
		return outputvec

	for i in xrange(len(doc)):
		outputvec += doc[i].vector

	return outputvec

def computeVectors(qu,an,img,VGGfeatures,nlp,img_map,encoder,nb_classes):
	features = np.zeros((1,4096+300))
	
	features[0,0:300] = question2VecSum(nlp,qu,300)
	features[0,300:] = VGGfeatures[:,img_map[img]]

	return features

def computeVectorsTimeSeries(qu,an,img,VGGfeatures,nlp,img_map,encoder,nb_classes, max_len):
	feature_dim = 300
	img_dim = 4096
	features_q = np.zeros((1, max_len, feature_dim))
	features_i = np.zeros((1, img_dim))

	features_q[0,:,:] = question2VecSum(nlp,qu,300)
	features_i[0,:] = VGGfeatures[:,img_map[img]]

	return features_i,features_q

def computeLanguageVectorsTimeSeries(qu,an,nlp,encoder,nb_classes, max_len):
	feature_dim = 300
	features_q = question2VecTimeSeries(nlp,qu,feature_dim, max_len)
	features_q = np.reshape(features_q, (1, features_q.shape[0], features_q.shape[1]))

	y = encoder.transform([an])
	Y = np_utils.to_categorical(y, nb_classes)

	return (features_q, Y)

def computeVectorsBatch(qu,an,img,VGGfeatures,nlp,img_map,encoder,nb_classes):
	word2vecDim = 300
	img_dim = VGGfeatures.shape[0]
	features = np.zeros((len(qu),img_dim+word2vecDim))
	for i in xrange(len(qu)):
		features[i,:word2vecDim] = question2VecSum(nlp,qu[i],word2vecDim)
		features[i,word2vecDim:] = VGGfeatures[:,img_map[img[i]]]

	y = encoder.transform(an)
	Y = np_utils.to_categorical(y, nb_classes)

	return (features, Y)

def computeLanguageVectorsBatch(qu,an,nlp,encoder,nb_classes):
	word2vecDim = 300
	features = np.zeros((len(qu),word2vecDim))
	for i in xrange(len(qu)):
		features[i,:word2vecDim] = question2VecSum(nlp,qu[i],word2vecDim)

	y = encoder.transform(an)
	Y = np_utils.to_categorical(y, nb_classes)

	return (features, Y)

def computeVectorsBatchTimeSeries(qu, an, img, VGGfeatures,img_map, nlp, encoder, nb_classes, max_len):
	feature_dim = 300
	img_dim = VGGfeatures.shape[0]
	
	features_q = np.zeros((len(qu), max_len, feature_dim))
	features_i = np.zeros((len(qu), img_dim))
	for i in xrange(len(qu)):
		features_q[i,:,:] = question2VecTimeSeries(nlp, qu[i], feature_dim, max_len)
		features_i[i,:] = VGGfeatures[:,img_map[img[i]]]

	y = encoder.transform(an)
	Y = np_utils.to_categorical(y, nb_classes)
	return (features_i, features_q, Y)