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
6. Caffe (Optional)

Tested with Python 2.7 on Ubuntu 14.04 and Centos 7.1.

**Notes**:

1. Keras needs the latest Theano, which in turn needs Numpy/Scipy. 
2. spaCy is currently used only for converting questions to a vector (or a sequence of vectors), this dependency can be easily be removed if you want to.
3. spaCy uses Goldberg and Levy's word vectors by default, but I found the performance to be much superior with Stanford's [Glove word vectors](http://nlp.stanford.edu/projects/glove/).
4. VQA Tools is **not** needed. 
5. Caffe (Optional) - For using the VQA with your own images.

##Installation Guide
This project has a large number of dependecies, and I am yet to make a comprehensive installation guide. In the meanwhile, you can use the following guide made by @gajumaru4444:

1. [Prepare for VQA in Ubuntu 14.04 x64 Part 1](https://gajumaru4444.github.io/2015/11/10/Visual-Question-Answering-2.html)
2. [Prepare for VQA in Ubuntu 14.04 x64 Part 2](https://gajumaru4444.github.io/2015/11/18/Visual-Question-Answering-3.html)

If you intend to use my pre-trained models, you would also need to replace spaCy's deafult word vectors with the GloVe word vectors from Stanford. You can find more details [here](http://spacy.io/tutorials/load-new-word-vectors/) on how to do this.

##Using Pre-trained models
Take a look at `scripts/demo_batch.py`. An LSTM-based pre-trained model has been released. It currently works only on the images of the MS COCO dataset (need to be downloaded separately), since I have pre-computed the VGG features for them. I do intend to add a pipeline for computing features for other images.

**Caution**: Use the pre-trained model with 300D Common Crawl Glove Word Embeddings. Do not the the default spaCy embeddings (Goldberg and Levy 2014). If you try to use these pre-trained models with any embeddings except Glove, your results would be **garbage**. You can find more deatails [here](http://spacy.io/tutorials/load-new-word-vectors/) on how to do this.

##Using your own images

Now you can use your own images with the `scripts/own_image.py` script. Use it like : 

python own_image.py --caffe /path/to/caffe

For now, a Caffe installation is required. However, I'm working on a Keras based VGG Net which should be up soon. Download the VGG Caffe model weights from [here](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) and place it in the scripts folder.

##The Numbers
Performance on the **validation set** and the **test-dev set** of the [VQA Challenge](http://visualqa.org/challenge.html):

| Model     		   | val           | test-dev      |
| ---------------------|:-------------:|:-------------:|
| BOW+CNN              | 48.46%		   | TODO		   |
| LSTM-Language only   | 44.17%        | TODO          |
| LSTM+CNN             | 51.63%        | 53.34%        |

Note: For validation set, the model was trained on the training set, while it was trained on both training and validation set for the test-dev set results.

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
