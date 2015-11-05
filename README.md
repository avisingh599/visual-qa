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

###Notes:
1. Keras needs the latest Theano, which in turn needs Numpy/Scipy. 
2. spaCy is currently used only for converting questions to a vector (or a sequence of vectors), this dependency can be easily be removed if you want to.
3. spaCy uses Goldberg and Levy's word vectors by default, but I found the performance to be much superior with Stanford's [Glove word vectors].

##The numbers
Performance on the **validation set** of the [VQA Challenge](http://visualqa.org/challenge.html):

| Model     		   | Accuracy      |
| ---------------------|:-------------:|
| BOW+CNN              | 44.30%		   |
| LSTM-Language only   | 42.51%        |
| LSTM+CNN             | 47.80%        |

There is a **lot** of scope for hyperparameter tuning here. Experiments were done for 100 epochs. 

| Model     		   | Training Time on GTX 760|
| ---------------------|:-----------------------:|
| BOW+CNN              | 160 seconds/epoch       |
| LSTM+CNN             | 200 seconds/epoch       |


##Get Started
Have a look at the `get_started.sh` script in the `scripts` folder. Also, have a look at the readme present in each of the folders.

##Feedback
All kind of feedback (code style, bugs, comments etc.) is welcome. Please open an issue on this repo instead of mailing me, since it helps me keep track of things better.

##License
MIT