Here is the utility of the various files:

0. `demo_batch.py`: You need access to pretrained models (included in the repo to run this example)

1. `get_started.sh`: Downloads data, VQAtools, pre-computed features, and trains a model. Run this script when you are done with the dependencies. 

2. `dumpText.py`: Dumps the questions and answers from the VQA json files to some text files for later ease of use. Run `python dumpText.py -h` for more info. 

3. `trainMLP.py`: Trains Multi-Layer perceptrons. Run `python trainMLP.py -h` for more info. 

4. `trainLSTM_1.py`: Trains LSTM-based model. Run `python trainLSTM_1.py -h` for more info. 

6. `trainLSTM_language.py`: Trains LSTM-based language-only model. Run `python trainLSTM_language.py -h` for more info. 

7. `evaluateMLP.py`: Evaluates models trained by `trainMLP.py`. Needs model json file, hdf5 weights file, and output txt file destinations to run.

8. `evaluateLSTM.py`: Evaluates models trained by `trainLSTM_1.py` and `trainLSTM_language.py`. Needs model json file, hdf5 weights file, and output txt file destinations to run.

9. `features.py`: Contains functions that are used to convert images and words to vectors (or sequences of vectors). 

10. `utils.py`: Exactly what you think.

11. `own_image.py`: Use your own image. Caffe installation required

12. `extract_features.py`: Extract 4096D VGG features from a VGG Caffe Model

13. `vgg_features.prototxt`: VGG Caffe Model Definition 
