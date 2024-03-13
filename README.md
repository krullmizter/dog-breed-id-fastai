# Dog Breed Identification built with Fast.ai's CNN using transfer learning

This was my thesis project for my bachelor's degree from Arcada University of Applied Sciences.

---
## Description

This notebook will take on a dog breed identification challenge by [Kaggle](https://www.kaggle.com/competitions/dog-breed-identification). The challenge uses the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), which is a subset of the much larger [ImageNet dataset](https://www.image-net.org/). This notebook also serves as a technical guide, and specification for developing an open-source dog breed identification model using Python and [Fast.ai's](https://github.com/fastai/fastai) Convolutional Neural Network (CNN) with transfer learning.

This notebook was created with the Stanford dataset in mind but the notebook can of course be used with other datasets as well. Feel free to modify the cells and code to fit your needs. This notebook serves as a guide and starting point for further development regarding Fast.ai and image classifications.

This notebook additionally explores the concepts of exploratory data analysis (EDA), data augmentation, image pre-processing, comprehensive logging of training statistics, the usage of libraries such as pandas, numpy, and matplotlib, the process of exporting, importing a trained model, and predicting new unseen images of dog breeds with the trained model.

This notebook also serves as the technical foundation for my bachelor's thesis on dog breed identification. The aim of this notebook, as well as my thesis, is to evaluate the efficiency and accuracy of my model when compared to similar models trained on the Stanford Dogs Dataset.

This notebook is quite extensively documented and uses various comments, and text cells to explain the development process. This notebook is not as detailed as my thesis, do read the thesis if you want a more in-depth understanding of the subjects brought up in this notebook. The combination of this notebook, and my thesis creates a unified "guide" to doing image classification using Fast.ai's CNN. Feel free to comment, critique, and create your version of this notebook.

- [Link to Thesis](https://www.theseus.fi/handle/10024/799064) 📖
- [Link to Github](https://github.com/krullmizter/dog-breed-id-fastai)

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
Begin by downloading or cloning the public repo [GitHub](https://github.com/krullmizter/dog-breed-id-fastai).

The GitHub repo contains an already created trained model directory and two dog breed images. The existing directory and images are not needed during the training or testing, but they've been kept in the public repo to ease the development for myself, and when it came to the evaluation of my bachelor thesis, for which this notebook is a part of. You can either delete both the images and the training folder when you start on your own or keep them as they are. 

**Remember to at least delete the `trained_model_stats.json` file in the training directory before you begin to log your stats, otherwise my personal training stats will be a part of your training stats file!**

### Setup
In the chapter **Settings, Variables & Paths** several changeable variables control if certain cells will run or not. Go over them with care before doing any training or testing.

### Dataset
This notebook will automatically download the Stanford dataset from my personal Google Drive, via a public link. But if you prefer you can [download](https://www.kaggle.com/competitions/dog-breed-identification/data) the Stanford dataset as a `.zip` file from Kaggle (you need a free Kaggle account to be able to download the file). 

**If you do download the `.zip` file yourself, or use a different dataset, be sure to upload the dataset `.zip` file to the root directory, and rename the file: `dataset.zip`.**

### Local Development
This notebook uses a multitude of different packages, libraries, and dependencies to make all the code in this notebook work. The repo contains two dependency files that you may use to get your development environment up and running. One for conda development environments `environment.yaml`, and one Python file for use in all development environments `requirements.txt`. The dependency files can be found in the GitHub [repo](https://github.com/krullmizter/dog-breed-id-fastai/tree/main/venv).

The Python dependencies are needed for both local development and everywhere else, like Conda, Google Colab, VS Code, Kaggle, and so on. The Python dependencies will be installed automatically in the **Installs & Imports** chapter using `pip`.

If you run this notebook locally, I recommend using Jupyter Notebook like [Anaconda Notebooks](https://anaconda.org/) and running the notebook with administrative privileges. The base Anaconda `environment.yaml` dependencies are only needed if you work with a conda environment. If you choose to do local development with Anaconda begin by creating a new conda environment and load the needed dependencies from the terminal. 

**The conda dependencies (environment.yaml) are different from the pip (requirements.txt)!**:

You need to be located in the directory of the `environment.yaml` to do this:
`conda env create -f environment.yaml` 

You can also create a new conda environment using the GUI called Anaconda Navigator. You can then import the `environment.yaml` file into your Anaconda navigator.

#### Errors
`PackagesNotFoundError`

If your conda installation can't find a certain package to download, then a tip is to use the dependency name, and the `-c` flag to specify from what channel you wish to download the dependency:

`conda install fastai pytorch pytorch-cuda -c fastai -c pytorch -c nvidia`

### Google Colab

If you want an easy way to run this notebook, use cloud-hosted GPUs, and have an easy time with dependencies and packages, then I recommend [Google Colab](https://colab.research.google.com/). To get started upload the `main.ipynb` to Colab. You can also clone my repo, upload it to your GitHub, and load your repo to Google Colab.

### Training Folder
When you run the `main.ipynb` file, with either the `export_model` or `log` variable set to true in the settings cell, a directory called `trained` will be created, in the root folder. It will hold a `.json` file with the stats of the model's training since its first successful training run. This way, one can view the past training stats to help with tweaking the model further. The directory will also hold the exported trained model as a `.pkl` file. It will also hold the exported .pkl file.

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
