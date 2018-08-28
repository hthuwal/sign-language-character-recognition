# sign-language-character-recognition
American Sign Language Character Recognition

The paper on this work is published [here](https://link.springer.com/chapter/10.1007/978-981-10-5547-8_42).

Do cite this paper, if you find it useful. :)

# Datset Used

The dataset can be found [here](http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html). Do cite their work if you plan to use their dataset.

# Setup and Usage
The repository contains the following two approaches to classify the sign language(ASL) characters (static).

## 1. Classification using SIFT and SVM
### 1.1 Setup 
To execute this process, make sure you have the following libraries installed :­
- matplotlib==1.5.
- networkx==1.1 1 
- numpy==1.11.
- Pillow==3.4.
- scikit­image==0.12.
- scipy==0.18.
- Sift==3.0.0.

The first step would be to make a folder structure as given below :­

```
SVM/
├── code
│   ├── classify.py
│   ├── grid.py
│   ├── learn.py
│   ├── libsvm
│   │   ├── svm-predict
│   │   ├── svm-scale
│   │   └── svm-train
│   ├── libsvm.py
│   ├── sift
│   └── sift.py
└── database
    ├── category0
    └── category1
```

#### code 

It contains all the code files and folders which are required to train and test the datasets.
  - `classify.py`   ­ 
    This file is required to classify the input image.
  - `gridy.py`   ­ 
    This file is required to perform cross validation on the training data.
  - `learn.py`   ­ 
    This file is required to train the svm model.
  - `sift.py`   ­ 
    This file is required to extract the features from the image.
  - `sift`   ­ 
    This is the executable of sift.py
  - `libsvm.py`   ­ 
    This file is required to tsin the model using svm algorithm.

#### database

It contains the images which are required to train the model. The database
folder should contains the folder name of the categories which you need to train. 

### 1.2 Training

For training of the model we need to run the following command :­
```
python learn.py
```

### 1.3 Testing

Run the following command to test the data set.
```
python classify.py pathToTestDataFolder
```

The folder structure of the test folder which contains the images to be classified will be same as like of the database folder.

The output after running the above command should be of the type :­
imageName ­­­­­> classification

## 2. Classification using CNN (Convolutional neural network)

### 2.1 Setup

#### Folder Structure
```
CNN/
├── commands.sh
├── extra_img.py
├── prediction.py
├── vgg5.py
├── database
|   ├── category0
|   └── category1
├── test
|   ├── category0
|   └── category1
└── weights
    └── vgg16_weights.h5

```

- `commands.sh`

    ```
    chmod +x commands.sh
    ./commands.sh
    ```
        
    This will install all the necessary dependencies.

- `vgg5.py` 
  
    VGG 16 model, training and testing.

- `extra_img.py` 
  
  - Reads and resizes each of the image to the similar size of 224x224  pixel.
  - Augment several images from each image using shear, zoom, horizontal and   vertical shifting.

- `prediction.py`   ­ 

  This file is required to predict a single image present in test folder.

- `database` 
  
  It contains the images which are required to train the model. The
  database folder should contains the folder name of the categories which you need to train

- `test` 
  
  This folder a single image which we need to predict after training
  the model.


- `weights`   ­
  
  It contains a pre trained weights based on VGG 16 model of upto 1 million images. The model file which we need to put in this folder can be obtained from the [link](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)
  
  *Note*: The name of the downloaded file should be *vgg16_weights.h5*


### 2.2 Training and Testing
Now run the following command.
```
python vgg5.py
```

The output would be a list of two floats. The first element will determine the loss and the second element specifies the accuracy.

#### Predicting a single image
Run the following command
```
python predict.py
```
The output will be the image and its prediction.

![](CNN/Example%20Predictions/2.png)
