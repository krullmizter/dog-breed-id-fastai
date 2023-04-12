# Dog Breed Identification built with Fast.ai's CNN using transfer learning
---
## Description

I recommend running this notebook with administrator privileges.

This project will take on a dog breed identification challenge by [Kaggle](https://www.kaggle.com/competitions/dog-breed-identification). The challenge uses the [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

You can download the entire dataset and labels as a .zip file (you need a free Kaggle account to be able to download the file). [Download dataset](https://www.kaggle.com/competitions/dog-breed-identification/data)

This project uses the [fast.ai](https://www.fast.ai/) library to create a convolutional neural network to classify dog breeds using transfer learning (ResNet34).

This project is the technical foundation of my bachelor's thesis regarding dog breed identification and the assessment of the speed and accuracy of my trained model against similar dog breed identification models presented in the Kaggle challenge.

This notebook also explores the concepts of exploratory data analysis (EDA) and other useful functions to ease the evaluation, testing and predicting capabilities of my trained model against.

---
## Structure

This is a broad overview of the main table of content for this notebook.
1.   Installs, Imports & Settings
2.   Load dataset
3.   EDA
4.   Training
5.   Dataloader
6.   Logging
7.   Post-Training Analysis
8.   Predictions
9.   Exports
10.   Clean-up
---
## Technical Specifications

This project was coded in a virtual environment using [anaconda notebooks](https://anaconda.org/). I recommend creating a separate development environment in Anaconda before starting off.

The successful training of a model will result in a directory called `training`. That directory will hold a JSON file with the stats of the model's training since its first successful training run, This way, one can view the past training stats to help with tweaking the model further. The directory will also hold the exported trained model as a `.pkl` file.

My training was computed locally on an RTX-3070 GPU.

Required installations and which versions I used (specified versions are not required):
* Python (3.10.9)
* PyTorch (2.0.0)
    * PyTorch CUDA (11.7)
* Fast.ai (2.7.12)

If you wish to use exactly my conda dependencies, and python packages then download and use the `environment.yml` and `requirements.txt` respectivly.

If your conda installation can't find a certain package to download, then a tip is to use the dependency name, and `-c` flag to specify from what channel you want to download the dependencies from like to code snippet below:

`conda install fastai pytorch pytorch-cuda -c fastai -c pytorch -c nvidia`

---
## TODO
* Automatic download of the Stanford dataset .zip file from Kaggle if it's not downloaded.
* Better: `item_tfms` and `batch_tfms`.
* Try to include more exceptions and better if statements.
* Single or multi-item detection.
* Add "time to train" to the validation metrics array
* Wrap certain cells, and longer code into easy to use functions and methods.
* View bounding boxes.
* Hover effect over the second scatter plot.
* Grammar, spell-checks, better comments, and structure.
* Clean up (remove .zip file a.s.o.).
* Link to thesis when done.
* Publish code, choose license.
---
Created by: Samuel Granvik [GitHub](https://github.com/krullmizter/) [LinkedIn](https://www.linkedin.com/in/samuel-granvik-93977013a/)