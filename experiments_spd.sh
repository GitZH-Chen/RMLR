#!/bin/bash

#---- SPDMLR and LieMLR on the HDM05_SPD ----

#---- SPDMLR ----
hdm05_path=/data #change this to your data folder

### Experiments on SPDNet
[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=LogEigMLR

### Experiments on SPDMLR-AIM
[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=SPDMLR\
  nnet.classifier.metric=AIM

### Experiments on SPDMLR-PEM
[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=SPDMLR\
  nnet.classifier.metric=PEM

[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=SPDMLR\
  nnet.classifier.metric=PEM nnet.classifier.power=0.5 nnet.classifier.beta=1/30

### Experiments on SPDMLR-LEM
[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=SPDMLR\
  nnet.classifier.metric=LEM

### Experiments on SPDMLR-BWM
[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=SPDMLR\
  nnet.classifier.metric=BWM nnet.classifier.power=0.5

### Experiments on SPDMLR-LCM
[ $? -eq 0 ] && python SPDMLR.py -m dataset=HDM05_SPD dataset.path=$hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.classifier.classifier=SPDMLR\
  nnet.classifier.metric=LCM nnet.classifier.power=1.,0.5