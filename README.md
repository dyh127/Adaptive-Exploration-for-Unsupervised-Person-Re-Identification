# [Adaptive-Exploration-for-Unsupervised-Person-Re-Identification](https://arxiv.org/pdf/1907.04194.pdf)([arxiv](https://arxiv.org/pdf/1907.04194.pdf))

![Framework of AE](https://github.com/dyh127/Adaptive-Exploration-for-Unsupervised-Person-Re-Identification/blob/master/images/framework.png)

## Prerequisites
* Python 3.6
* Pytorch 1.0
## Datasets
1. Create folder to save data ```mkdir data```.
2. Download the [datasets](https://drive.google.com/drive/folders/1gP_-NPynQct5APKF55cg2NwfmuE8kpT-?usp=sharing) (Market-1501, DukeMTMC-reID and MSMT17). If you want dataset from Baidu Yun, please refer to [ECN](https://github.com/zhunzhong07/ECN) (Thanks to [Zhun Zhong](http://zhunzhong.site/)).
3. Unzip them and put the unzipped file under ```data/```.
4. The data structure would look like:
```
data/
    market/
          bounding_box_train/
          bounding_box_test/
          bounding_box_train_camstyle/
          query/
    duke/
          bounding_box_train/
          bounding_box_test/
          bounding_box_train_camstyle/
          query/
    msmt17/
          bounding_box_train/
          bounding_box_test/
          bounding_box_train_camstyle/
          query/
```
## Train
run ```bash train.sh```.
## Test
run ```bash test.sh```.

## Results (paper)
1. Market1501(market) and DukeMTMC-reID(duke)

|**Method & data**|**Map**|**rank-1**|**rank-5**|**rank10**|
|:---:|:---:|:---:|:---:|:---:|
|duke to market|58.0%|81.6%|91.9%|94.6%|
|market only|54.0%|77.5%|89.8%|93.4%|
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
@article{journals/tomccap/DingFXY20,
  author    = {Yuhang Ding and Hehe Fan and Mingliang Xu and Yi Yang},
  title     = {Adaptive Exploration for Unsupervised Person Re-Identification},
  journal   = {{TOMM}},
  volume    = {16},
  number    = {1},
  pages     = {3:1--3:19},
  year      = {2020},
  doi       = {10.1145/3369393},
}
```
## Related Repos
https://github.com/zhunzhong07/ECN

https://github.com/Cysu/open-reid

https://github.com/layumi/Person_reID_baseline_pytorch

