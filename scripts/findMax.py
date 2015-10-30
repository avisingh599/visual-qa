from spacy.en import English
import numpy as np
questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
#questions_val = open('../data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
questions = questions_train

nlp = English()
len_max = 0
question_lengths = []
for q in questions:
	len_cur = len(nlp(q))
	question_lengths.append(len_cur)

np.save('question_lengths',np.asarray(question_lengths))

print len_max

