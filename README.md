# [Adaptive-Exploration-for-Unsupervised-Person-Re-Identification](https://arxiv.org/pdf/1907.04194.pdf)



## Prerequisites
* Python 3.6
* Pytorch 1.0
## Datasets
1. Create folder to save data ```mkdir data```.
2. Download the datasets Market-1501 and DukeMTMC-reID.
3. Unzip them and put the unzipped file under ```data/```.
4. The data structure would look like:
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
1. Create folder to save model ```mkdir checkpoint```.
2. Download the model (market2duke and duke2market).
3. Unzip them and put the unzipped file under ```checkpoint/```.
4. run ```bash test.sh```.

## Citation
If you find the code useful, considering citing our work:
```
@article{DBLP:journals/corr/abs-1907-04194,
  author    = {Yuhang Ding and
               Hehe Fan and
               Mingliang Xu and
               Yi Yang},
  title     = {Adaptive Exploration for Unsupervised Person Re-Identification},
  journal   = {CoRR},
  volume    = {abs/1907.04194},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.04194},
  archivePrefix = {arXiv},
  eprint    = {1907.04194},
  timestamp = {Wed, 17 Jul 2019 10:27:36 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1907-04194},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
