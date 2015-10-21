import operator
import argparse
import sys
import os

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

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-isTrain', type=int, default=1)
	args = parser.parse_args()

	if args.isTrain == 1:
		annFile = '../data/Annotations_Train_mscoco/mscoco_train2014_annotations.json'
		quesFile = '../data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json'
		questions_file = open('../data/preprocessed/questions_train2014.txt', 'w')
		answers_file = open('../data/preprocessed/answers_train2014.txt', 'w')
		coco_image_id = open('../data/preprocessed/images_train2014.txt', 'w')
	else:
		annFile = '../data/Annotations_Val_mscoco/mscoco_val2014_annotations.json'
		quesFile = '../data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json'
		questions_file = open('../data/preprocessed/questions_val2014.txt', 'w')
		answers_file = open('../data/preprocessed/answers_val2014.txt', 'w')
		coco_image_id = open('../data/preprocessed/images_val2014.txt', 'w')

	#initialize VQA api for QA annotations
	vqa=VQA(annFile, quesFile)
	questions = vqa.questions
	ques = questions['questions']
	qa = vqa.qa;

	for i, q in zip(xrange(1,len(ques)+1),ques):
		questions_file.write(q['question'].encode('utf8'))
		questions_file.write('\n'.encode('utf8'))
		coco_image_id.write(str(q['image_id']).encode('utf8'))
		coco_image_id.write('\n')
		if args.isTrain:
			answers_file.write(getModalAnswer(qa[i]['answers']).encode('utf8'))
		else:
			answers_file.write(getAllAnswer(qa[i+248349]['answers']).encode('utf8'))
		answers_file.write('\n'.encode('utf8'))

	print 'completed dumping', qora