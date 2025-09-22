# COMIL 


[Tingting Zheng](https://scholar.google.com/citations?user=AJ5zl-wAAAAJ&hl=zh-CN), [Hongxun Yao](https://scholar.google.com/citations?user=aOMFNFsAAAAJ),
[Kui jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl=en&oi=ao), [Yi Xiao](https://scholar.google.com/citations?user=e3a4aG0AAAAJ).

 


**Abstract:** Effective instance representation and bag prediction are critical in multi-instance learning (MIL) for histopathology whole slide image (WSI) analysis. Most current methods focus on elaborating either the instance aggregators or bag predictors, while neglecting the synergistic promotion between these aspects during optimization. To mitigate this gap, we proposed COMIL, a unified framework that jointly enhances instance representation and bag prediction through collaborative learning. COMIL employs an instance fuser to capture correlations among instances, a neighbor contrastive learning module to further derive reliable and generalized representations, and a transformer-based aggregator to integrate high-quality features for robust bag prediction. By jointly optimizing these components within a multi-task learning paradigm, COMIL effectively explores and leverages the mutual information between representation and prediction, thus improving overall optimization quality. Extensive experiments on four benchmark datasets demonstrate that COMIL outperforms state-of-the-art methods, achieving accuracy improvements of 1.2\% and 3.3\% over the pseudo-bag-based DTFD, and 4.5\% and 2.0\% over the bag-based ACMIL, on the CAMELYON16 and TCGA-ESCA datasets, respectively, highlighting the effectiveness of COMIL in WSI  classification tasks.


## Update
- [2025/09/22] Uploading code

## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on 3090)

## Dependencies:
```bash
torch
torchvision
numpy
h5py
scipy
scikit-learning
pandas
nystrom_attention
admin_torch
```


The data used for training, validation and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──DATASET_1_DATA_DIR/
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
        └── h5_files
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
    ├──DATASET_2_DATA_DIR/
        └── pt_files
                ├── slide_a.pt
                ├── slide_b.pt
                └── ...
        └── h5_files
                ├── slide_i.h5
                ├── slide_ii.h5
                └── ...
    └── ...
```

## Train COMIL

Split the dataset.
```
python datasetsplitting.py --PATH_LABEL_CSV 
```

Training, validation, and testing of COMIL:
```
python mainfile.py --PATH_LABEL_CSV  --h5, csv, result,
```
