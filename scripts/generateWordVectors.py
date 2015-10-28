from gensim.models import Word2Vec
import regex as re

def preprocess_text(text):
	text = text.lower()
	return re.sub(ur"\p{P}+", " ", text).split()

def main():
	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	sentences = []
	for questions in questions_train:
		sentences.append(preprocess_text(questions))

	model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4)
	model.save('../features/questionstrain2014_word_vector.bin')

if __name__ == "__main__":
	main()
