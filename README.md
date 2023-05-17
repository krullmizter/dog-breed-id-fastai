# Dog Breed Identification built with Fast.ai's CNN using transfer learning
---
## Description

This notebook will take on a dog breed identification challenge by [Kaggle](https://www.kaggle.com/competitions/dog-breed-identification). The challenge uses the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), which is a subset of the much larger [ImageNet dataset](https://www.image-net.org/). This notebook also serves as a technical guide, and specification for developing an open-source dog breed identification model using Python and [Fast.ai's](https://github.com/fastai/fastai) Convolutional Neural Network (CNN) with transfer learning.

This notebook can of course be used with other datasets as well, feel free to modify the cells and code to fit your needs. This notebook servers as a guide and starting point for further development.

This notebook additionally explores the concepts of exploratory data analysis (EDA), data augmentation, image pre-processing, comprehensive logging of training statistics, the usage of libraries such as pandas, numpy and matplotlib among others, and the process of exporting and importing a trained model.

This project also serves as the technical foundation for my bachelor's thesis on dog breed identification. The aim of this notebook, as well as my thesis, is to evaluate the efficiency and accuracy of my model when compared to similar models trained on the Stanford Dogs Dataset.

This notebook is quite extensively documented, and uses various comments, and text cells to explain the development process. This notebook is not as detailed as my thesis, but the combination of this notebook, and my thesis creates a unified "guide" to doing image classification using Fast.ai. Feel free to comment, critique, and create your own version of this notebook.

[Link to Thesis]() 📖

[Link to Github repo](https://github.com/krullmizter/dog-breed-id-fastai)

---
## Goals

The goal of an image classification problem is to minimize the loss. Loss refers to the measure of how well a model's predictions match the actual classes/labels of the training data. A lower loss value indicates that the model is more accurate at making predictions.

Striving for a high level of accuracy is also key. Accuracy is measured by how well the trained model can correctly predict the classes of unseen new images.

---
## Structure

This is a broad overview of the main table of contents of this notebook:
1.   Installs, Imports & Settings
2.   Load the dataset
3.   EDA
4.   Dataloader
5.   Training
6.   Logging
7.   Post-Training Analysis
8.   Predictions
9.   Export
10.  Import trained model
---
## Technical Specifications

### Setup
Begin by downloading or cloning this projects public repo [GitHub](https://github.com/krullmizter/dog-breed-id-fastai).

You can download and use the base environment files: `environment.yaml`, `requirements.txt` for conda, and Python respectively. The files can be found in the [repo](https://github.com/krullmizter/dog-breed-id-fastai/tree/main/venv).

In the chapter **Settings, Variables & Paths** there are several changeable variables that controls if certain cells will run or not. Go over them with care before doing any training or testing.

### Dataset
This notebook will automatically download the Stanford dataset from my personal Google Drive, via a public link. But if you prefer you can [download](https://www.kaggle.com/competitions/dog-breed-identification/data) the Stanford dataset as a `.zip` file from Kaggle (you need a free Kaggle account to be able to download the file). If you do download the `.zip` file yourself, or use a different dataset, be sure to upload the dataset `.zip` file to the root directory, and rename the file: `dataset.zip`.

### Local Development
If you run this notebook locally, I recommend using Jupyter Notebook like [Anaconda notebooks](https://anaconda.org/), creating a new environment, and running the notebook with administrative privileges.

Create a new conda environment from the terminal:

`conda env create -f environment.yaml` or import the `environment.yaml` file into your Anaconda navigator.

Install all the base Python packages with pip:

`pip install -r requirements.txt`

#### Errors
`PackagesNotFoundError`

If your conda installation can't find a certain package to download, then a tip is to use the dependency name, and the `-c` flag to specify from what channel you wish to download the dependency from:

`conda install fastai pytorch pytorch-cuda -c fastai -c pytorch -c nvidia`

### Google Colab

If you want an easy way to run this notebook, use cloud-hosted GPUs, and have an easy time with dependencies and packages, then I recommend [Google Colab](https://colab.research.google.com/). To get started upload the `main.ipynb` to Colab. You can also clone my repo, and upload it to your GitHub and load your own repo to Google Colab. 

### Training Folder
When running this notebook, with either the `export_model` or `log` variable set to true in the settings cell, a directory called `trained` will be created, in the root folder. It will hold a `.json` file with the stats of the model's training since its first successful training run. This way, one can view the past training stats to help with tweaking the model further. The directory will also hold the exported trained model as a `.pkl` file. It will also hold the exported .pkl file.

### My Development

My training was computed locally on an RTX-3070 GPU using Anaconda, and online via [Google Colab](https://colab.research.google.com/).

The main software and libraries I used (specified versions are not required):
* Anaconda (1.11.1)
    * Conda (23.3.1)
* Python (3.10.9)
    * pip (22.3.1)
* PyTorch (2.0.0)
    * PyTorch CUDA (11.7)
* Fast.ai (2.7.12)

---
## TODO
* View bounding boxes.
* Hover effect over the second scatter plot.
* Link to thesis when done.
---

## Copyright 

Copyright (C) 2023 Samuel Granvik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

---
This code was created by Samuel Granvik. If you use or modify this code, please give attribution to Samuel Granvik. 

Links: [Email](samgran@outlook.com) | [GitHub](https://github.com/krullmizter/) | [LinkedIn](https://www.linkedin.com/in/samuel-granvik-93977013a/)

---

<p>My dog Laban ❤️</p>
<img src='https://github.com/krullmizter/dog-breed-id-fastai/blob/main/laban.jpg?raw=1' width='10%' height='10%' >
