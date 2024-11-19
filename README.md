# Model Implosion -- Compute Alignment Difficulty (AD)
This is the code for CCS '24 paper "Understanding Implosion in Text-to-Image Generative Models" to compute AD.

## Overview
Code to accompany: "Understanding Implosion in Text-to-Image Generative Models" [paper](https://arxiv.org/abs/2409.12314).

---

## Requirements

The code uses a virtual environment, with python 3.10. To set it up, run
```
$ python3 -m venv venv_implosion
$ source venv_implosion/bin/activate
$ pip3 install torch==2.1 torchvision==0.16
$ pip3 install git+https://github.com/openai/CLIP.git
$ pip3 install open-clip-torch==2.24.0
$ pip3 install einops==0.7.0
$ pip3 install numpy==1.26.4
```

---

## Running the code

We provide code to compute the Alignment Difficulty for a given text-image dataset. By default, the code computes the AD for data from the LAION Aesthetics dataset, scaling the feature cosine distance as detailed in the paper. But it also supports running experiments on other datasets when formated as text-image pairs (e.g., leveraging labels or generated prompts as text for classification datasets like CIFAR10 and ImageNet). Below, we provide general information on setting up the codebase. 

### Datasets
- The LAION Aesthetics dataset can be downloaded from [here](https://laion.ai/blog/laion-aesthetics/).
- The ImageNet dataset can be downloaded from [here](https://image-net.org/download.php).
- The CIFAR10 dataset can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). 


You may need to reformat the dataset such that each text-image pair is saved as a pickle file, formatted as
```
{ 'img': [array] image,
  'text': [str] text }
```

### Computing AD on a dataset

To compute the AD on a dataset, run
```
python3 compute_AD.py -data_glob_path 'PATH_TO_DATASET/*.pickle' -save_emb_path 'PATH_SAVE_EMB/filename.pickle' -alpha 0.8
```

The code will glob files from ```data_glob_path```, which should be edited according to how the dataset is stored, e.g., if the pickle files are in subdirectories organized by class, change the argument value to ```PATH_TO_DATASET/*/*.pickle```. 

For each pickle file of text-image pair, normalized CLIP embeddings will be computed, and saved in a pickle file at ```save_emb_path```. The normalization eases computation for cosine similarity, which is used in computing AD. Finally, the code will return the AD value for the dataset using the trade-off parameter $\alpha$ in the argument, which defaults to $\alpha=0.8$ if not provided. 

---

## Citation

```
@inproceedings{ding2024understand,
  title={Understanding Implosion in Text-to-Image Generative Models},
  author={Ding, Wenxin and Li, Cathy Y and Shan, Shawn and Zhao, Ben Y and Zheng, Haitao},
  journal={Proc. of CCS},
  year={2024}
}
```