# DeepMirTar
Stacked denoising autoencoder (SdA) code of "Deep learning based functional site-level and UTR-level human miRNA target prediction". The code was rewritten from www.deeplearning.net 


<br>

The usage of the SdA is similar to sklean:


## pseudo-example:

from SdA_wm import SdA

sda_classifier = SdA()

sda_classifier.pretraining(train_x) 

sda_classifier.finetuning(train_x, train_y, valid_x, valid_y)    # the valid set is used to optimize the parameters

y_pred = sda_classifier.predict(test_y)

>>>More detaild example, see test_SdA.py

<br>

## Dependencies:
1), python 2.7, latest version

2), theano, latest version

<br>

## Further reading: 

1), A collection of all the existing miRNA target gene prediction papers and some miRNA related databases.

https://github.com/Bjoux2/Existing-miRNA-target-gene-prediction-papers

2), Deep-Learning-in-Bioinformatics-Papers-Reading-Roadmap

https://github.com/Bjoux2/Deep-Learning-in-Bioinformatics-Papers-Reading-Roadmap
