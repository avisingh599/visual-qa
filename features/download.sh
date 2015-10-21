#!/bin/bash
# Downloads and unzips the VGG features computed on the COCO dataset. 

wget http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip
unzip coco.zip -d .