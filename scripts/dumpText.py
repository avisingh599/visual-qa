import operator
import argparse
import sys
import os
import progressbar

from spacy.en import English

if os.path.isdir('../3rdParty/VQA/PythonHelperTools/vqaTools/'):
	sys.path.insert(0, '../3rdParty/VQA/PythonHelperTools/vqaTools/')
else:
	print 'Please download the VQA tools and out them in the 3rdParty folder'
from vqa import VQA

def getModalAnswer(answers):
	candidates = {}
	for i in xrange(10):
		candidates[answers[i]['answer']] = 1

	for i in xrange(10):
		candidates[answers[i]['answer']] += 1

	return max(candidates.iteritems(), key=operator.itemgetter(1))[0]

def getAllAnswer(answers):
	answer_list = []
	for i in xrange(10):
		answer_list.append(answers[i]['answer'])

	return ';'.join(answer_list)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-isTrain', type=int, default=1)
	args = parser.parse_args()

	nlp = English() #used for conting number of tokens

	if args.isTrain == 1:
		annFile = '../data/mscoco_train2014_annotations.json'
		quesFile = '../data/OpenEnded_mscoco_train2014_questions.json'
		questions_file = open('../data/preprocessed/questions_train2014.txt', 'w')
		questions_lengths_file = open('../data/preprocessed/questions_lengths_train2014.txt', 'w')
		answers_file = open('../data/preprocessed/answers_train2014.txt', 'w')
		coco_image_id = open('../data/preprocessed/images_train2014.txt', 'w')
		trainval = 'training data'
	else:
		annFile = '../data/mscoco_val2014_annotations.json'
		quesFile = '../data/OpenEnded_mscoco_val2014_questions.json'
		questions_file = open('../data/preprocessed/questions_val2014.txt', 'w')
		questions_lengths_file = open('../data/preprocessed/questions_lengths_val2014.txt', 'w')
		answers_file = open('../data/preprocessed/answers_val2014.txt', 'w')
		coco_image_id = open('../data/preprocessed/images_val2014.txt', 'w')
		trainval = 'validation data'

	#initialize VQA api for QA annotations
	vqa=VQA(annFile, quesFile)
	questions = vqa.questions
	ques = questions['questions']
	qa = vqa.qa

	
	pbar = progressbar.ProgressBar()
	print 'Dumping questions,answers, imageIDs, and questions lenghts to text files...'
	for i, q in pbar(zip(xrange(1,len(ques)+1),ques)):
		questions_file.write(q['question'].encode('utf8'))
		questions_file.write('\n'.encode('utf8'))
		questions_lengths_file.write(str(len(nlp(q['question']))).encode('utf8'))
		questions_lengths_file.write('\n'.encode('utf8'))

		coco_image_id.write(str(q['image_id']).encode('utf8'))
		coco_image_id.write('\n')
		if args.isTrain:
			answers_file.write(getModalAnswer(qa[q['question_id']]['answers']).encode('utf8'))
		else:
			answers_file.write(getAllAnswer(qa[q['question_id']]['answers']).encode('utf8'))
		answers_file.write('\n'.encode('utf8'))

	print 'completed dumping', trainval

if __name__ == "__main__":
	main()