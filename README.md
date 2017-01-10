# DeepMirTar
Stack denoising autoencoder (SdA) code of "Deep learning based functional site-level and UTR-level human miRNA target prediction"
The usage of the SdA is similar to sklean:

e.g．

sda_classifier = SdA()

sda_classifier.pretraining(train_x) # unsupervised pretraning

sda_classifier.finetuning(train_x, train_y, valid_x, valid_y) # superinvised traning, the valid set is used to 

                                                              # optimize the parameters

y_pred = sda_classifier.predict(test_y)

Dependencies:

1), python 2.7, latest version

2), theano, latest version
