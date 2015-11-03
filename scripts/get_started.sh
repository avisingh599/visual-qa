#!/bin/sh
#make other scripts executable
chmod +x ../data/download.sh
chmod +x ../3rdParty/download.sh
chmod +x ../features/download.sh
#download data and features
cd ../data/
./download.sh
cd ../3rdParty/
./download.sh
cd ../features/
./download.sh

#convert the data to a simpler format
cd ../scripts
python dumpText.py
python dumpText.py -isTrain 0

#train and evaluate models
python trainMLP.py
python evaluateMLP.py -model ../models/mlp_num_hidden_units_1024_num_hidden_layers_3.json -weights ../models/mlp_num_hidden_units_1024_num_hidden_layers_3_epoch_00_loss_5.10.hdf5 -results ../results/mlp_1024_3_ep0.txt
