# [Adaptive-Exploration-for-Unsupervised-Person-Re-Identification](https://arxiv.org/pdf/1907.04194.pdf)

![Framework of AE](https://github.com/dyh127/Adaptive-Exploration-for-Unsupervised-Person-Re-Identification/blob/master/images/framework.png)

## Prerequisites
* Python 3.6
* Pytorch 1.0
## Datasets
1. Create folder to save data ```mkdir data```.
2. Download the [datasets](https://drive.google.com/drive/folders/1gP_-NPynQct5APKF55cg2NwfmuE8kpT-?usp=sharing) (Market-1501 and DukeMTMC-reID).
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
1. Create folder to save trained model ```mkdir checkpoint```.
2. Download the [trained model](https://drive.google.com/drive/folders/1w54U0QNIvE8PDEryAa_5jYZxt0UhnhAh?usp=sharing) (market2duke and duke2market).
3. Unzip them and put the unzipped file under ```checkpoint/```.
4. run ```bash test.sh```.

## Results (paper)
1. Market1501(market) and DukeMTMC-reID(duke)

|**Method & data**|**Map**|**rank-1**|**rank-5**|**rank10**|
|:---:|:---:|:---:|:---:|:---:|
|duke to market|58.0%|81.6%|91.9%|94.6%|
|market only|54.0%|77.5%|89.8%|93.4|
|market to duke|46.7%|67.9%|79.2%|83.6%|
|duke only|39.0%|63.2%|75.4%|79.4%|
2. MSMT17(msmt17)

|**Method & data**|**Map**|**rank-1**|**rank-5**|**rank10**|
|:---:|:---:|:---:|:---:|:---:|
|market to msmt17|9.2%|25.5%|37.3%|42.6%|
|duke to msmt17|11.7%|32.3%|44.4%|50.1%|
|msmt17 only|8.5%|26.6%|37.0%|41.7%|

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
## Related Repos
