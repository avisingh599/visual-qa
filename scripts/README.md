Here is the utility of the various files

0. `get_started.sh`: For a sample run of an experiment.

1. `dumpText.py`: Dumps the questions and answers from the VQA json files to some text files for later ease of use. By default, it works only the training set, pass the parameter `isTrain 0` to dump the validation data. 

2. `trainMLP.py`: Trains Multi-Layer perceptrons. Run `python trainLSTM.py -h` for more info. 

3. `trainLSTM.py`: Trains LSTM-based model. Run `python trainLSTM.py -h` for more info. 

4. `trainLSTM_language.py`: Trains LSTM-based language-only model. Run `python trainLSTM_language.py -h` for more info. 

5. `evaluateMLP.py`: Evaluates models trained by `trainMLP.py`. Needs model json file, hdf5 weights file, and output txt file destinations to run.

6. `evaluateLSTM.py`: Evaluates models trained by `trainMLP.py`. Needs model json file, hdf5 weights file, and output txt file destinations to run.

7. `features.py`: Contains functions that are used to convert images and words to vectors (or sequences of vectors). 

8. `utils.py`: Exactly what you think.