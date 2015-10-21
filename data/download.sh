#!/bin/bash
# Downloads the training and validation sets from visualqa.org. 

wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip

unzip \*.zip