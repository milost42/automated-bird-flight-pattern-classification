# automated-bird-flight-pattern-classification
Extraction and classification of bird flight patterns from video. Accompanying paper:

## Installation
```git clone https://github.com/milost42/automated-bird-flight-pattern-classification.git```

## Install dependencies
Tested versions:
sklearn-image 0.22.0
sklearn-learn 1.2.1
matplotlib 3.9.2
numpy 1.26.1
pandas 2.1.1
opencv-python 4.9.0.80
opencv-contrib-python 4.9.0.80
torch 2.1.0+cu121
detecto 1.2.2
joblib 1.3.2
moviepy 2.0.0.dev2

## Datasets
Arrange the datasets in the Datasets folder structure

[Other flying objects](https://zenodo.org/records/18281801)

[M2 training dataset](https://zenodo.org/records/18281931)

[M2 full-frame test dataset](https://zenodo.org/records/18282125)

## Description of files
**generate_5_sec_clip_dataset.py**: Creates a dataset of 5-second clips from the videos in the Final video set folder. 

**generate_balanced_dataset.py**: Creates a balanced dataset from the 5-second clips generated, with the same number of clips (40) per species. 

**generate_test_dataset.py**: Creates a test dataset of 5-second clips, with 4 clips per species. To run this code, first run the folder_set_up function and then manually select clips for each species from the 
balanced dataset that have sections in which the bird is not in the frame. Place these clips in the "non" folder for each species. Label each frame with bird and non in the generated Excel sheet under the column title Bird actual and upstroke, downstroke and non under the column title Motion actual. 

**bird_object_main.py**: This is the main file that contains an object with all the functions needed for classification. 

**m1.py**: This file runs Model 1 (bird/non). 

**m2.py**: This file runs Model 2 (upstroke/downstroke/non). 

**generate_patterns.py**: Generates the flight patterns for a video from the Model 2 predictions.

**analyse_patterns.py**: Analyses a flight pattern to get the features for species classification. 

**get_species.py**: Predicts the species from the features of a flight pattern using an example of Model 3. 

**all_tests.py**: This file has all the tests that generated the figures in the corresponding paper. 

**generate_features_balanced_dataset.py**: Generate the features of the flight patterns for all the clips in the balanced dataset. 

**cross_validation_species.py**: Cross-validate the species identification using the features of the flight patterns for the balanced dataset clips. 

**model2_weights_final.py**: Weights for Model 2

**model3_best.joblib**: Best example of weights for a species identification model

## Model training

M2 was trained using the training dataset linked above and with transfer learning through the detecto module. With all the labels and corresponding images in the same folder (e.g. 'images/'), training can be easily acheived with the following (provided GPU capability is enabled):

```
from detecto import core, utils, visualize

dataset = core.Dataset('images/')
model = core.Model(['downstroke', 'upstroke'])

model.fit(dataset)
```
