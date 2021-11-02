# CNN Based Segmentation of Infarcted Regions in Acute Cerebral Stroke Patients From Computed Tomography Perfusion Imaging (mJ-Net)

## Release v1.0
It contains the code described in the paper "CNN Based Segmentation of Infarcted Regions in Acute Cerebral Stroke Patients From Computed Tomography Perfusion Imaging"

### 1 - Abstract
More than 13 million people suffer from ischemic cerebral stroke
worldwide each year. Thrombolytic treatment can reduce brain
damage but has a narrow treatment window. Computed Tomography
Perfusion imaging is a commonly used primary assessment
tool for stroke patients, and typically the radiologists will evaluate
resulting parametric maps to estimate the affected areas, dead tissue
(core), and the surrounding tissue at risk (penumbra), to decide
further treatments. Different work has been reported, suggesting
thresholds, and semi-automated methods, and in later years deep
neural networks, for segmenting infarction areas based on the parametric
maps. However, there is no consensus in terms of which
thresholds to use, or how to combine the information from the
parametric maps, and the presented methods all have limitations
in terms of both accuracy and reproducibility.

![alt text](images/mjnetdet_2.png?raw=true)

### 1.1 - Link to paper

- ACM Library : https://doi.org/10.1145/3388440.3412470
- arxiv.org: https://arxiv.org/abs/2104.03002


### 2 - Dependecies:
```
pip install -r requirements.txt
```

### 3 - Usage
Assuming that you already have a dataset to work with, you can use a json file to define the setting of your model.

Refer to  [Setting_explained.json](Setting/Setting_explained.json) for explanations of the various settings.


### 3.1 Train/Test

```
Usage: python main.py gpu sname
                [-h] [-v] [-d] [-o] [-s SETTING_FILENAME] [-t TILE] [-dim DIMENSION] [-c {2,3,4}]

    positional arguments:
      gpu                   Give the id of gpu (or a list of the gpus) to use
      sname                 Select the setting filename

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase output verbosity
      -d, --debug           DEBUG mode
      -o, --original        Set the shape of the testing dataset to be compatible with the original shape
                            (T,M,N) [time in front]
      -pm, --pm             Set the flag to train the parametric maps as input
      -t TILE, --tile TILE  Set the tile pixels dimension (MxM) (default = 16)
      -dim DIMENSION, --dimension DIMENSION
                            Set the dimension of the input images (width X height) (default = 512)
      -c {2,3,4}, --classes {2,3,4}
                            Set the # of classes involved (default = 4)
      -w, --weights         Set the weights for the categorical losses

```


### 4 - How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper when you have used it in your study.
```
@inproceedings{tomasetti2020cnn,
  title={CNN Based Segmentation of Infarcted Regions in Acute Cerebral Stroke Patients From Computed Tomography Perfusion Imaging},
  author={Tomasetti, Luca and Engan, Kjersti and Khanmohammadi, Mahdieh and Kurz, Kathinka D{\ae}hli},
  booktitle={Proceedings of the 11th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},
  pages={1--8},
  year={2020}
}
```

### Got Questions?
Email me at luca.tomasetti@uis.no
