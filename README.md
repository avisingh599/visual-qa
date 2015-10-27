#Deep Learning for Visual Question Answering

This project uses Keras to train a variety of **Feedforward** and **Recurrent Neural Networks** for the task of Visual Question Answering. It is designed to work with the [VQA](http://visualqa.org) dataset. 


##Requirements
1. Keras 0.20
2. spaCy 0.94
3. scikit-learn 0.16
4. Python 2.7

###Notes:
1. Keras needs the latest Theano, which in turn needs Numpy/Scipy. 
2. Also, if you want GPU acceleration (which is definitely needed when running on the entire VQA dataset), then you must install the CUDA and the appropriate drivers for your GPU.
3. spaCy is currently used only for converting questions to a vector (or a sequence of vector), this dependency can be easily be removed if you want to.

