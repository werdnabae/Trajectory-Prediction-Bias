# Discovering and Understanding Algorithmic Biases in Autonomous Pedestrian Trajectory Predictions

Andrew Bae, Susu Xu

This repo contains the code for our workshop paper: [Discovering and Understanding Algorithmic Biases in Autonomous Pedestrian Trajectory Predictions](https://dl.acm.org/doi/pdf/10.1145/3560905.3568433), published in [The Fourth Workshop on Continual and Multimodal Learning for Internet of Things (CML-IOT 2022)](https://cmliot2022.github.io/) at [SenSys 2022](https://sensys.acm.org/2022/).

# Environment
Our code was implemented on a desktop computer with an NVIDIA GeForece RTX 3080 10GB. 

Install the conda environment from the yml file: 

```
conda env create -f environment.yml
```

## Data Preparation
The [JAAD](https://github.com/ykotseruba/JAAD) and [PIE](https://github.com/aras62/PIE) datasets can be downloaded from their respective websites. Follow their directions to extract the necessary files. To obtain the [TITAN](https://usa.honda-ri.com/titan) dataset, refer to this [page](https://usa.honda-ri.com/titan) and contact the authors directly.



## Training
We used the checkpoints provided by the BiTraP authors for the JAAD and PIE datasets. 

Checkpoints for SGNet on JAAD and PIE used to be made availible by the authors, but it seems like their linked folder no longer contain any checkpoints. 

We trained the models on the TITAN dataset from scratch. 

We provide all of our utilized checkpoints in the checkpoints folder.

Training commands if you would like to train the models yourself:

###

## Testing
A .pkl file will be generated after each test. This will be used later for the data analysis.

### BiTraP

### SGNet

## Analysis
Analysis was done using various Jupyter notebooks. They have been provided in the "notebooks" folder.

## Citation
```
@inproceedings{10.1145/3560905.3568433,
author = {Bae, Andrew and Xu, Susu},
title = {Discovering and Understanding Algorithmic Biases in Autonomous Pedestrian Trajectory Predictions},
year = {2023},
isbn = {9781450398862},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3560905.3568433},
doi = {10.1145/3560905.3568433},
booktitle = {Proceedings of the 20th ACM Conference on Embedded Networked Sensor Systems},
pages = {1155–1161},
numpages = {7},
keywords = {trajectory prediction, bias, fairness, algorithm evaluation},
location = {Boston, Massachusetts},
series = {SenSys '22}
}
```