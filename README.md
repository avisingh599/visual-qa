#Deep Learning for Visual Question Answering

This project uses Keras to train a variety of **Feedforward** and **Recurrent Neural Networks** for the task of Visual Question Answering. It is designed to work with the [VQA](http://visualqa.org) dataset. 

Models Implemented:

1. A Feedforward Model
2. An LSTM-based model

##Requirements
1. Keras 0.20
2. spaCy 0.94
3. scikit-learn 0.16
4. progressbar
5. Nvidia CUDA 7.5 (optional, for GPU acceleration)

Tested on Python 2.7 on Ubuntu 14.04 and Centos 7.1.

###Notes:
1. Keras needs the latest Theano, which in turn needs Numpy/Scipy. 
2. spaCy is currently used only for converting questions to a vector (or a sequence of vectors), this dependency can be easily be removed if you want to.

##Get Started
Have a look at the `get_started.sh` script in the `scripts` folder. Also, have a look at the readme present in each of the folders.

##License
MIT