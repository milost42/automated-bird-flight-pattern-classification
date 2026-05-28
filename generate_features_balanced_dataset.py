import os
import pandas as pd
from bird_object_main import Bird
import numpy as np

"""
Filename: generate_features_balanced_dataset.py
Date: 22-01-2026
Version: 1.0.1
Description: Extract the flight patterns and their features from all the clips in the 5-second clips balanced dataset
"""


def create_excel_sheet(balanced_clips_excel_address, features_excel_address):
    """Create Excel sheet of features for each clip in the balanced dataset"""

    # Retrieve columns from Excel sheet
    balanced_clips_dataset = pd.read_excel(balanced_clips_excel_address).to_numpy()
    vids = balanced_clips_dataset[:, 0]
    species = balanced_clips_dataset[:, 1]
    groups = balanced_clips_dataset[:, 2]

    # Create a DataFrame from the above columns with an additional features column populated with zeros
    features_dataset = pd.DataFrame({"Video": vids, "Species": species, "Group": groups,
                                     "Features": np.zeros(len(vids))})

    # Write the DataFrame to a new Excel sheet
    features_dataset.to_excel(features_excel_address, header=True, index=False)


def generate_features(excel_address, index):
    """Generate the features for a single clip and write them to the corresponding row for the clip in the Excel
    sheet"""

    # Read the Excel sheet and convert to numpy array
    features_dataset = pd.read_excel(excel_address)
    features_dataset_arr = features_dataset.to_numpy()

    # Retrieve columns from Excel sheet
    vids = features_dataset_arr[:, 0]
    labels = features_dataset_arr[:, 1]
    groups = features_dataset_arr[:, 2]
    vid_features = features_dataset_arr[:, 3]

    print(labels)
    print(groups)
    print(vids)

    # Create an empty string for the features
    features_string = ""

    # Create a new Bird object with the address of the clip
    # The address is retrieved from the vids column by using the input index to select the row
    bird_object = Bird(vids[index])

    # Extract all the flight patterns from the clip and get the features for each of them
    all_features, sec_per_frame = bird_object.get_features()
    print(all_features)

    # Convert all the feature sets into a string to write to the Excel sheet
    if len(all_features) > 0:
        for features in all_features:
            for feature in features:
                # A comma is used to separate each feature within a set
                features_string += str(feature) + ","
            # A semicolon is used to denote the end of a feature set
            features_string += ";"
    else:
        # If there are no flight patterns extracted and therefore no features, set an error message
        features_string = "no flight pattern generated"

    print(features_string)
    print("\n")

    # Write the features string to the correct row for the input clip
    vid_features[index] = features_string
    print(vid_features)

    # Write the features column back to the DataFrame
    features_dataset['Features'] = vid_features

    # Write the Dataframe back to the Excel sheet
    features_dataset.to_excel(excel_address, index=False)


# Change "path/to" to the path to the git folder
address = "path/to/automated-bird-flight-pattern-classification/Datasets/Dataset excel sheets/"

# Set the addresses for the balanced clip dataset and the new features dataset
balanced_vids_excel_address = address + "5 second clips balanced.xlsx"
features_excel_address = address + "5 second clips features.xlsx"

# RUN ONLY ONCE THE FIRST TIME
# create_excel_sheet(balanced_vids_excel_address, features_excel_address)

# Get number of videos
df = pd.read_excel(features_excel_address)
vid_no = len(df["Video"])
print(vid_no)

# Cycle through all the videos in dataset
for index in range(0, vid_no):
    generate_features(features_excel_address, index)