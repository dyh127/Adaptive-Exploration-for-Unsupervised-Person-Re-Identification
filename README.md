# [Adaptive-Exploration-for-Unsupervised-Person-Re-Identification](https://arxiv.org/pdf/1907.04194.pdf)



## Prerequisites
* Python 3.6
* Pytorch 1.0
## Datasets
1. Download the datasets Market-1501 and DukeMTMC-reID.
2. Unzip them and put the unzipped file under ```data/```.
3. The data structure would look like:
```
data/
    market/
          bounding_box_train/
          bounding_box_test/
          query/
    duke/
          bounding_box_train/
          bounding_box_test/
          query/
```
## Test
(training code is still being collated. sorry for that)
1. Download the model (market2duke and duke2market).
2. Unzip them and put the unzipped file under ```checkpoint/```.
3. run ```bash test.sh```.
