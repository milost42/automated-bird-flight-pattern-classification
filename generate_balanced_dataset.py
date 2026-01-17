import os
from random import randint
from moviepy.editor import *
import math
import shutil
import pandas as pd

"""
Filename: generate_balanced_dataset.py
Author: Mili Ostojic
Date: 09-01-2026
Version: 1.0
Description: Generate a dataset with a balanced number of clips per species from the 5-second clips dataset
"""


def generate_balanced_dataset(address):
    """Generate a dataset of 5-second clips with a balanced number of clips for each species"""

    # Get the addresses of all the species folders in the original video set
    original_vid_folders = [address + "Final video set/" + folder + "/" for folder in
                            os.listdir(address + "Final video set/")]
    print(original_vid_folders)

    # Set addresses for 5-second clips dataset and balanced clips dataset
    clip_address = address + "5 second vids/"
    balanced_clip_address = address + "5 second vids balanced/"

    # Check if the balanced clips dataset already exists
    clips_folder_exist = os.path.isdir(balanced_clip_address)
    print(clips_folder_exist)

    if not clips_folder_exist:
        # If balanced clips dataset does not exist, create the folder
        os.mkdir(balanced_clip_address)
    else:
        # If balanced clips dataset does exist, remove all the clips from it
        files = os.listdir(balanced_clip_address)
        for file in files:
            file_path = os.path.join(balanced_clip_address, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    test_vids = []

    # Cycle through species folders
    for folder in original_vid_folders:

        # Get the species name from the folder name
        species_name = folder.split("/")[-2]

        # Define the address of the species sub-folder in the balanced clips dataset
        balanced_species_folder = balanced_clip_address + species_name
        print(balanced_species_folder)

        # Check if the species sub-folder exists
        balanced_species_folder_exist = os.path.isdir(balanced_species_folder)
        if not balanced_species_folder_exist:
            # If species sub-folder does not exist, create it
            os.mkdir(balanced_species_folder)
        else:
            # If species sub-folder does exist, remove all clips from it
            files = os.listdir(balanced_species_folder)
            for file in files:
                file_path = os.path.join(balanced_species_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        species_test_vids = []

        # Get list of original videos
        videos = os.listdir(folder)

        # Cycle through until there are 40 chosen clips for the species
        # 40 is the max number of videos a species folder has (40 Red Kite, 23 Kestrel, 10 Black-headed gull,
        # 7 Sparrowhawk)
        while len(species_test_vids) < 40:

            # Cycle through the original videos
            for vid in videos:

                skip_vid = False

                # Get the name of the current original video
                vid_name = vid.split("/")[-1].split(".")[0]
                print(vid_name)

                # Open the current original video and get the duration
                clip = VideoFileClip(folder + vid)
                duration = clip.duration

                # Check if the current original video has any associated 5-second clips
                if duration >= 5:

                    # Calculate total number of 5-second clips generated from the current original video
                    total_clip_no = math.trunc((duration - 5) / 2) + 1

                    # Select a random 5-second clip from the current original video
                    random_index = randint(0, total_clip_no - 1)
                    path = clip_address + species_name + "/" + vid_name + "_" + str(random_index) + ".mp4"

                    chosen_indexes = []

                    # Select another random clip from the current original video if the chosen one has already been
                    # selected
                    while path in species_test_vids:

                        # Select a random 5-second clip from the current original video
                        random_index = randint(0, total_clip_no - 1)
                        path = clip_address + species_name + "/" + vid_name + "_" + str(random_index) + ".mp4"

                        print(chosen_indexes)
                        print(set(chosen_indexes))

                        # Check if all the available clips for the current original video have already been attempted
                        if len(set(chosen_indexes)) == total_clip_no:
                            # If there are no more clips that haven't already been attempted, skip the current video
                            print("SKIP VID")
                            skip_vid = True
                            break

                        # Update the list of clips that have been attempted for the current original video
                        chosen_indexes.append(random_index)
                else:
                    # Skip the video as all the video is less than 5 seconds long so there are no available clips
                    skip_vid = True
                    print("SKIP VID")

                # If the current video has not been skipped and there are less than 40 clips selected, update the list
                # of selected clips with the chosen clip
                if not skip_vid and len(species_test_vids) < 40:
                    species_test_vids.append(path)
                    print(len(species_test_vids))
                    print("CONTINUE")

        print(len(species_test_vids))
        print(len(set(species_test_vids)))

        print("SPECIES DONE")

        # Update the overall list of chosen clips with the clips chosen for the current species
        for chosen_vid in species_test_vids:
            test_vids.append(chosen_vid)

    # Copy the selected clips from the clips dataset to the balanced clip dataset
    for vid_path in test_vids:
        folder_name = vid_path.split("/")[-2]
        vid_name = vid_path.split("/")[-1]

        print(balanced_clip_address + folder_name + "/" + vid_name)

        shutil.copy2(vid_path, balanced_clip_address + folder_name + "/" + vid_name)


def generate_excel_sheet(address, excel_address):
    """Generate an Excel spreadsheet for the balanced dataset"""

    # Check if the folder of dataset Excel sheets exists
    excel_folder_exist = os.path.isdir(address + "Dataset excel sheets/")

    # Create the dataset Excel sheets folder if it does not exist
    if not excel_folder_exist:
        os.mkdir(address + "Dataset excel sheets/")

    # Get the full paths of all the species folders in the balanced dataset
    species_folders = [address + "5 second vids balanced/" + folder + "/" for folder in
                       os.listdir(address + "5 second vids balanced/")]
    print(species_folders)

    # Get the full paths of all the clips in the balanced dataset
    all_vids = []
    for folder in species_folders:
        print(folder)
        vids = [folder + vid for vid in os.listdir(folder)]

        for vid in vids:
            all_vids.append(vid)

    print(all_vids)

    # Initialise variables
    groups = []
    original_vids = []
    group = 0
    species = []

    # Get a list of "groups" for the chosen clips, referring to the original video the clip came from

    # Cycle through all chosen clips except the final one
    for i in range(len(all_vids)-1):

        # Get the filename of the original video the current clip came from
        vid = all_vids[i]
        original_vid = vid.split(".")[0].removesuffix("_" + vid.split(".")[0].split("_")[-1]) + ".mp4"

        # Get the filename of the original video the next clip came from
        next_vid = all_vids[i+1]
        next_original_vid = next_vid.split(".")[0].removesuffix("_" + next_vid.split(".")[0].split("_")[-1]) + ".mp4"

        # Update the list of species with the species of the current clip
        species.append(vid.split("/")[-2])

        # Check if the original video the current clip came from has already been added (which happens when a section
        # of clips originating from the same original video is complete)
        if original_vid not in original_vids:

            # Update the list of groups with the group corresponding to the current original video
            groups.append(group)

            # Check if the next clip comes from the same original video as the current one
            if original_vid != next_original_vid:
                # If they are from different videos, there are no more clips originating from the current original video

                # Update the list of original videos
                original_vids.append(original_vid)
                # Increment the group
                group += 1

    # # Get the original video for the final clip
    final_vid = all_vids[-1].split(".")[0].removesuffix("_" + all_vids[-1].split(".")[0].split("_")[-1]) + ".mp4"

    # Update the list of original videos with the final original video
    # If it is the same as for the previous clip, this completes the section of clips originating from the same video
    # If it is not the same, then the new original video also has to be added
    original_vids.append(final_vid)

    # Update the list of groups with the group corresponding to the final clip
    # If it is the same as for the previous clip, the group won't have been incremented in the for loop
    # If it is different to the previous clip, the group will have been incremented in the for loop
    groups.append(group)

    # Update the list of species with the current species
    species.append(all_vids[-1].split("/")[-2])

    print(original_vids)
    print(groups)
    print(species)

    # Create a dictionary with all the clips and their corresponding species and original video groups
    data = {'Video': all_vids, 'Species': species, 'Group': groups}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the Dataframe to an Excel sheet
    df.to_excel(excel_address, header=True, index=False)


# Change "path/to" to the path to the git folder
address = "path/to/automated-bird-flight-pattern-classification/Datasets/"

# Generate balanced dataset of 5-second clips
generate_balanced_dataset(address)

# Generate Excel sheet for the balanced dataset of 5-second clips
excel_address = address + "Dataset excel sheets/5 second clips balanced new.xlsx"
generate_excel_sheet(address, excel_address)
