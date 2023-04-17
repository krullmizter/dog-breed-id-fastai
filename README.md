# Dog Breed Identification built with Fast.ai's CNN using transfer learning
---
## Description

This project will take on a dog breed identification challenge by [Kaggle](https://www.kaggle.com/competitions/dog-breed-identification). The challenge uses the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

You can [download](https://www.kaggle.com/competitions/dog-breed-identification/data) the entire dataset as a `.zip` file (you need a free Kaggle account to be able to download the file).

This project employs the [Fast.ai](https://github.com/fastai/fastai) library to create an image classification model that leverages transfer learning and a convolutional neural network (CNN) to accurately identify different dog breeds.

This project serves as the technical foundation for my bachelor's thesis on dog breed classification. The aim of this project, as well as my thesis, is to evaluate the efficiency and accuracy of my model when compared to similar models trained on the Standford Dogs Dataset.

This notebook additionally explores the concepts of exploratory data analysis (EDA), data augmentation, and image pre-processing among others.

---
## Goals

The goal of an image classification problem is to minimize the loss. Loss refers to the measure of how well a model's predictions match the actual classes/labels of the training data. A lower loss value indicates that the model is more accurate at making predictions.

Striving for a high level of accuracy is also key. Accuracy is measured by how well the trained model can correctly predict the classes of unseen new images.

---
## Structure

This is a broad overview of the main table of contents for this notebook:
1.   Installs, Imports & Settings
2.   Load the dataset
3.   EDA
4.   Training
5.   Dataloader
6.   Logging
7.   Post-Training Analysis
8.   Predictions
9.   Exports
10.  Clean-up
---
## Technical Specifications

Begin by downloading the repo [GitHub](https://github.com/krullmizter/dog-breed-identification-fastai). If you don't have the dataset `.zip` file, download it from [Kaggle](https://www.kaggle.com/competitions/dog-breed-identification/data) (A free Kaggle account is needed).

### Local Development

If you run this notebook locally, I recommend running it with administrative privileges.

This project was coded locally in a virtual environment using [Anaconda notebooks](https://anaconda.org/). When working with Anaconda I recommend creating a separate development environment before starting. You can use my base env. file `environment.yml` from the GitHub repo.

### Google Colab

If you want an easy way to run this notebook, use cloud-hosted GPUs, and have an easy time with dependencies and packages, then use [Google Colab](https://colab.research.google.com/). To get started upload the `main.ipynb` to Colab. Then upload the dataset `.zip` file to your Google Drive. Lastly, change `colab = True` in the settings cell.

### Training Stats

When working with this notebook, a directory called `training` will be created. It will hold a `.json` file with the stats of the model's training since its first successful training run. This way, one can view the past training stats to help with tweaking the model further. The directory will also hold the exported trained model as a `.pkl` file.

### Development

My training was computed locally on an RTX-3070 GPU.

Required installations and which versions I used (specified versions are not required):
* Python (3.10.9)
* PyTorch (2.0.0)
    * PyTorch CUDA (11.7)
* Fast.ai (2.7.12)

If you wish to use exactly my dependencies, and Python packages then download and use the `environment.yml` and `requirements.txt` respectively from the repo [GitHub](https://github.com/krullmizter/dog-breed-identification-fastai).

If your conda installation can't find a certain package to download, then a tip is to use the dependency name, and the `-c` flag to specify from what channel you wish to download the dependency from:

`conda install fastai pytorch pytorch-cuda -c fastai -c pytorch -c nvidia`

---
## TODO
* Better: `item_tfms` and `batch_tfms`.
* Single or multi-item detection.
* View bounding boxes.
* Hover effect over the second scatter plot.
* Link to thesis when done.
* Publish code, choose a license.
---
Created by: Samuel Granvik [GitHub](https://github.com/krullmizter/) | [LinkedIn](https://www.linkedin.com/in/samuel-granvik-93977013a/)