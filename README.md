## Analtics Vidhya - Game of Deep Learning

### Problem:
The objective was to classify maritime vessels images into 5 different categories a.k.a. Cargo, Tanker, Millitary, Cruise and ___.

Roughly 6K images were provided for training with uneven distribution aross classes. Test set has around 2K images wth distribution of classes similar to test set.

Weighted F1 was used to measure performance of models.


### Approach:

Using trasnfer learning for this competition was obvious choice. I decided to use fast.ai v1 library for this competition as it had many best practices for transfer learning incorporated in it.


Per literature survey and my past experience, I have found resnet50, inceptionv4 and densnet169 trained on imagenet to perform really well on transfer learning.


The plan was to try out different image augmenetations and training/finetuning schedules to reach optimal accuracy for different models and then average out predictions.


### Solution:

1. Densnent169 - after trying out few different traning schedules, I settled on following:
    a.) train head with fit one cycle for 5 epochs and 3e-4 lr
    b.) unfeeze all layers
    c.) fit one cyelc over 5 epochs 4 times with lr's 5e-4, 1e-4, 5e-5, 1e-5 ___


2. Inceptionv4 - For inception fitting head prior to unfreezing all layers was resulting in very slow convergence, so I decided to unfreeze all layeres in first step and followed same trainig schedule as in 1.

3. I split training into 8 stratified folds and ran model for two folds to make predictions robust. 


4. Inception model gave a slightly lower validation so I decided to give 0.7 and 0.3 weights to probabilities from densenet and inception models respectively for final submission.


### Failed ideas  

1. I tried changing default values of augmentations, but didnt notice any change in accuracy so decided to keep them at defaults

2. I tried fitting rectangular images as mostly provided images were rectangular, again it didn't result in any improvement.


### Reproducing solution:

1. git clone repo
2. Make change in config.py to reflect correct train, test and images file paths
3. Run `bash run_all.sh` in bash shell

### Dependencies
  *  fastai == ___

