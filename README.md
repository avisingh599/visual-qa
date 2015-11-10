#Deep Learning for Visual Question Answering

[Click here](https://avisingh599.github.io/deeplearning/visual-qa/) to go to the accompanying blog post. 

This project uses Keras to train a variety of **Feedforward** and **Recurrent Neural Networks** for the task of Visual Question Answering. It is designed to work with the [VQA](http://visualqa.org) dataset. 

Models Implemented:

|BOW+CNN Model  |  LSTM + CNN Model |
|--------------------------------------|-------------------------| 
| <img src="https://raw.githubusercontent.com/avisingh599/homepage/master/images/vqa/model_1.jpg" alt="alt text" width="400" height=""> | <img src="https://raw.githubusercontent.com/avisingh599/homepage/master/images/vqa/lstm_encoder.jpg" alt="alt text" width="300" height="whatever"> |


##Requirements
1. [Keras 0.20](http://keras.io/)
2. [spaCy 0.94](http://spacy.io/)
3. [scikit-learn 0.16](http://scikit-learn.org/)
4. [progressbar](https://pypi.python.org/pypi/progressbar)
5. Nvidia CUDA 7.5 (optional, for GPU acceleration)

Tested with Python 2.7 on Ubuntu 14.04 and Centos 7.1.

**Notes**:
1. Keras needs the latest Theano, which in turn needs Numpy/Scipy. 
2. spaCy is currently used only for converting questions to a vector (or a sequence of vectors), this dependency can be easily be removed if you want to.
3. spaCy uses Goldberg and Levy's word vectors by default, but I found the performance to be much superior with Stanford's [Glove word vectors](http://nlp.stanford.edu/projects/glove/).

##Using Pre-trained models
Take a look at `scripts/demo_batch.py`. An LSTM-based pre-trained model has been released. It currently works only on the images of the MS COCO dataset (need to be downloaded seperately). I do intend to add a pipeline for it to work for all images in general.
### Caution: Use the pre-trained model with 300D Common Crawl Glove Word Embeddings
Do not the word embeddings which are the default spaCy embeddings (Goldberg and Levy 2014). If you try to use these models with any embeddings except Glove, your results woulf be **garbage**. 

##The Numbers
Performance on the **validation set** of the [VQA Challenge](http://visualqa.org/challenge.html):

| Model     		   | Accuracy      |
| ---------------------|:-------------:|
| BOW+CNN              | 44.30%		   |
| LSTM-Language only   | 42.51%        |
| LSTM+CNN             | 47.80%        |

There is a **lot** of scope for hyperparameter tuning here. Experiments were done for 100 epochs. 

Training Time on various hardware:

| Model     		   | GTX 760             |  Intel Core i7      |
| ---------------------|:-------------------:|:-------------------:|
| BOW+CNN              | 140 seconds/epoch   | 900 seconds/epoch   |
| LSTM+CNN             | 200 seconds/epoch   | 1900 seconds/epoch  |

The above numbers are valid when using a batch size of `128`, and training on 215K examples in every epoch.

##Get Started
Have a look at the `get_started.sh` script in the `scripts` folder. Also, have a look at the readme present in each of the folders.

##Feedback
All kind of feedback (code style, bugs, comments etc.) is welcome. Please open an issue on this repo instead of mailing me, since it helps me keep track of things better.

##License
MIT