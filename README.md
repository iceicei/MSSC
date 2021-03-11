## MSSC

The code for our paper Improving Pedestrian Attribute Recognition With Multi-Scale Spatial Calibrate.


## Dataset Info

PA100K[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.pdf)][[Github](https://github.com/xh-liu/HydraPlus-Net)]

RAP : A Richly Annotated Dataset for Pedestrian Attribute Recognition
- v1.0 [[Paper](https://arxiv.org/pdf/1603.07054v3.pdf)][[Project](http://www.rapdataset.com/)]
- v2.0 [[Paper](https://ieeexplore.ieee.org/abstract/document/8510891)][[Project](http://www.rapdataset.com/)]


## Get Started
1. Run `git clone https://github.com/iceicei/MSSC.git`
2. Create a directory to dowload above datasets.
    ```
    cd MSSC
    ```
3. Prepare datasets to have following structure:
    ```
    ${project_dir}/data
        PA100k
            data/
            annotation.mat
            README.txt
        RAP
            RAP_dataset/
            RAP_annotation.mat
            README.txt
        RAP2
            RAP_dataset/
            RAP_annotation.mat
            README.txt
    ```
4. Run the `format_xxxx.py` to generate `dataset.pkl` respectively
    ```
    python ./dataset/preprocess/format_pa100k.py
    python ./dataset/preprocess/format_rap.py
    python ./dataset/preprocess/format_rap2.py
    ```
5. Train MSSC
    ```
    python train.py PA100k
    ```



### Citation

If you use this method or this code in your research, please cite as:




